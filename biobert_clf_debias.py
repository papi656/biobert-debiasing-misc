import os 
import math 
import pandas as pd
import argparse
import earlyStopping
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F
import loss_function
from loss_function import *

input_path = 'datasets'
output_path = 'resources'

MAX_LEN = 310 # suitable for all datasets
MAX_GRAD_NORM = 10
BATCH_SIZE = 6
LEARNING_RATE = 1e-5


def read_data(dataset_name):
    train_path = os.path.join(input_path, dataset_name, 'train.txt')
    devel_path = os.path.join(input_path, dataset_name, 'devel.txt')
    train_token_lst, train_label_lst = [], []
    with open(train_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                train_token_lst.append(math.nan)
                train_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            train_token_lst.append(a[0].strip())
            train_label_lst.append(a[1].strip())

    train_data = pd.DataFrame({'Tokens': train_token_lst, 'Labels': train_label_lst})

    devel_token_lst, devel_label_lst = [], []
    with open(devel_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                devel_token_lst.append(math.nan)
                devel_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            devel_token_lst.append(a[0].strip())
            devel_label_lst.append(a[1].strip())

    devel_data = pd.DataFrame({'Tokens': devel_token_lst, 'Labels': devel_label_lst})

    return train_data, devel_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if isinstance(tok, float):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    return sent_list,label_list

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    sentence = str(sentence).strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(',')):
        # tokenize and count num of subwords
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        labels.extend([label]*n_subwords)

    return tokenized_sentence, labels 

def loadBiasProb(dataset_name, bias_file):
    bias_file_path = os.path.join('bias_probs', dataset_name, bias_file)
    bias_probs = []
    probs_per_sent = []
    with open(bias_file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                if len(probs_per_sent) > 0:
                    bias_probs.append(probs_per_sent)
                    probs_per_sent = []
            else:
                parts = line.split(',')
                tmp_probs = []
                max_pos, max_val = 0, float('-inf')
                to_subtract = 0
                for i, p in enumerate(parts):
                    tmp_probs.append(float(p))
                    if float(p) > max_val:
                        max_val = float(p)
                        max_pos = i
                    if float(p) == 0:
                        tmp_probs[i] = 0.001
                        to_subtract += 0.001

                tmp_probs[max_pos] -= to_subtract
                probs_per_sent.append(tmp_probs)

    return bias_probs

def loadTeacherProb(dataset_name, model_name):
    prob_file_name = 'train_softmax_' + model_name + '.txt'
    prob_file_path = os.path.join('resources', dataset_name, prob_file_name)
    teacher_probs = []
    probs_per_sent = []
    with open(prob_file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                if len(probs_per_sent) > 0:
                    teacher_probs.append(probs_per_sent)
                    probs_per_sent = []
            else:
                parts = line.split(',')
                tmp_probs = []
                max_pos, max_val = 0, float('-inf')
                to_subtract = 0
                for i, p in enumerate(parts):
                    tmp_probs.append(float(p))
                    if float(p) > max_val:
                        max_val = float(p)
                        max_pos = i
                    if float(p) == 0:
                        tmp_probs[i] = 0.001
                        to_subtract += 0.001

                tmp_probs[max_pos] -= to_subtract
                probs_per_sent.append(tmp_probs)

    if len(probs_per_sent) > 0:
        teacher_probs.append(probs_per_sent)

    return teacher_probs


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label
        self.maximum_across_all = 0 

    def __getitem__(self, index):
        # step 1: tokenize sentence and adapt labels
        sentence = self.data.Sentence[index]
        word_labels = self.data.Labels[index]
        label2id = self.label2id

        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens and corresponding labels
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        labels.insert(0, 'O')
        labels.insert(-1, 'O')
        
        # step 3: truncating or padding
        max_len = self.max_len

        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
            labels = labels[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]
            labels = labels + ['O' for _ in range(max_len - len(labels))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.int32),
            'mask': torch.tensor(attn_mask, dtype=torch.int32),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

class bertWithCustomLoss(nn.Module):
    def __init__(self, model_dir, num_labels, id2label, label2id, loss_fn:ClfDebiasLossFunction):
        super(bertWithCustomLoss, self).__init__()
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        self.id2label = id2label
        self.label2id = label2id
        self.biobert = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id)

    def forward(self, num_labels, input_ids, attention_mask, labels=None, bias=None, teacher_probs=None):
        outputs = self.biobert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # logits = outputs[0]
        # active_pos = attention_mask.view(-1) == 1
        # active_logits = logits.view(-1, self.num_labels)
        # active_logits = torch.masked_select(active_logits, active_pos)
        if bias is not None:
            loss = self.loss_fn.forward(num_labels, outputs.hidden_states[-1], outputs[0], bias, teacher_probs, labels)
        else:
           lossFunction = nn.CrossEntropyLoss()
           loss = lossFunction(outputs[0].view(-1,self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=outputs[0],
            hidden_states=outputs.hidden_states
        )
    
    def save_model(self, output_path):
        self.biobert.save_pretrained(output_path)

def train(model, dataloader, optimizer, device, sent_lst, tokenizer, bias_probability, teacher_probability):
    tr_loss, tr_accuracy = 0, 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
 
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.int32)
        mask = batch['mask'].to(device, dtype=torch.int32)
        targets = batch['targets'].to(device, dtype=torch.long)

        bias_prob = []
        teacher_prob = []
        b_no = 0
        for index in indexes:
            sentence = sent_lst[index]
            words = sentence.split(' ')
            bias_prob_per_sent = []
            teacher_prob_per_sent = []
            bias_prob_per_sent.append([0.001, 0.001, 0.998]) # for [CLS] token
            if teacher_probability is not None:
                teacher_prob_per_sent.append([0.001, 0.001, 0.998])
            for i, word in enumerate(words):
                subwords = tokenizer.tokenize(word)
                for _ in range(len(subwords)):
                    bias_prob_per_sent.append(bias_probability[index][i])
                    if teacher_probability is not None:
                        if i < len(teacher_probability[index]):
                            teacher_prob_per_sent.append(teacher_probability[index][i])
                        else:
                            teacher_prob_per_sent.append([0.001, 0.001, 0.998])
            bias_prob_per_sent.append([0.001, 0.001, 0.998]) # for [SEP] token
            if teacher_probability is not None:
                teacher_prob_per_sent.append([0.001, 0.001, 0.998])
            for m in mask[b_no]: # for [PAD] tokens
                if m == 0:
                    bias_prob_per_sent.append([0.001, 0.001, 0.998])
                    if teacher_probability is not None:
                        teacher_prob_per_sent.append([0.001, 0.001, 0.998])
            b_no += 1
            bias_prob.append(bias_prob_per_sent)
            if teacher_probability is not None:
                teacher_prob.append(teacher_prob_per_sent)
        # print(len(bias_prob), len(bias_prob[0]), len(bias_prob[0][0]))
        bias_prob = torch.tensor(bias_prob, dtype=torch.float16)
        # print(bias_prob.shape)
        bias_prob = bias_prob.to(device, dtype=torch.float16)
        if teacher_probability is not None:
            teacher_prob = torch.tensor(teacher_prob, dtype=torch.float16)
            teacher_prob = teacher_prob.to(device, dtype=torch.float16)

        outputs = model(model.num_labels, input_ids, mask, labels=targets, bias=bias_prob, teacher_probs=teacher_prob)

        loss = outputs.loss
        tr_logits = outputs.logits

        tr_loss += loss.item()
        tr_logits = F.softmax(tr_logits, dim=2)
        nb_tr_steps += 1
        if idx%100 == 0:
            print(f'\tTraining loss at {idx} steps: {tr_loss}')

        #compute training accuracy
        flattened_targets = targets.view(-1)
        active_logits = tr_logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, dim=1)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_preds.extend(predictions)
        # tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\tTraining loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')


def valid(model, dataloader, device):
    eval_loss = 0
    model.eval()

    for batch in dataloader:
        input_ids = batch['ids'].to(device, dtype=torch.int32)
        mask = batch['mask'].to(device, dtype=torch.int32)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(model.num_labels, input_ids, mask, labels=targets)

        loss = outputs.loss
        eval_loss += loss.item()

    return eval_loss 



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--loss_fn', type=str, required=True)
    parser.add_argument('--bias_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    # read data
    train_data, devel_data = read_data(args.dataset_name)
    
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    num_labels = len(id2label)

    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    devel_sent,devel_label = convert_to_sentence(devel_data)

    bias_probs = loadBiasProb(args.dataset_name, args.bias_file)
    teacher_probs = None 
    
    #load tokenizer
    tokenizer_dir = os.path.join('resources', args.dataset_name, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    # tokenizer = AutoTokenizer.from_pretrained('/home/abhishek/Desktop/MTP/biobert_ft_final/resources/BC5CDR/tokenizer')

    # setting the loss function
    if args.loss_fn == 'BiasProduct':
        print(f'\n _____Bias Product______\n')
        loss_fn = loss_function.BiasProduct()
    elif args.loss_fn == 'Learned-Mixin':
        print(f'\n _____Learned-Mixin+H______\n')
        loss_fn = loss_function.LearnedMixinH(0.03)
    elif args.loss_fn == 'Reweight':
        print(f'\n _____Reweight______\n')
        loss_fn = loss_function.Reweight()
    elif args.loss_fn == 'Confidence_regularization':
        print(f'\n _____Confidence Regularization______\n')
        teacher_probs = loadTeacherProb(args.dataset_name, args.model_name)
        loss_fn = loss_function.SmoothedDistillLoss()
    else:
        print(f'\n _____Plain cross entropy______\n')
        loss_fn = loss_function.Plain()
        
    model_dir = os.path.join('resources', args.dataset_name, args.model_name)
    # model = bertWithCustomLoss.from_pretrained(model_dir, num_labels, id2label, label2id, loss_fn=loss_fn)
    model = bertWithCustomLoss(model_dir, num_labels, id2label, label2id, loss_fn)

    device = 'cuda' if cuda.is_available() else 'cpu'
    #loading model to device
    model.to(device)

    train_data = {'Sentence':train_sent, 'Labels':train_label}
    train_data = pd.DataFrame(train_data)
    devel_data = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_data = pd.DataFrame(devel_data)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False
                    }

    devel_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True
                    }
    
    train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)
    devel_dataset = dataset(devel_data, tokenizer, MAX_LEN, label2id, id2label)
    devel_dataloader = DataLoader(devel_dataset, **devel_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    num_epochs = 30 # no reason, IEEEAccess paper used this, so trying with this number

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        train(model, train_dataloader, optimizer, device, train_sent, tokenizer, bias_probs, teacher_probs)
        validation_loss = valid(model, devel_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')

    m_name = args.model_name + '_' + args.loss_fn
    model_path = os.path.join(output_path, args.dataset_name, m_name)
    model.save_model(model_path)
    # model_path = model_path + '/model.pt'
    # model.save_pretrained(model_path)
    # torch.save(model.state_dict(), model_path)
    # Save the model's configuration separately
    # config_path = os.path.join(output_path, args.dataset_name, m_name, 'config')
    # model.config.save_pretrained(config_path)


if __name__ == '__main__':
    main()
