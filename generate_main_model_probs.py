import os 
import math 
import pandas as pd
import argparse
import earlyStopping
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F


input_path = 'datasets'
output_path = 'resources'


MAX_LEN = 310
BATCH_SIZE = 6

def read_data(dataset_name):
    train_path = os.path.join(input_path, dataset_name, 'train.txt')
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

    return train_data

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

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label

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
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len


def get_softmax_output(model, dataloader, tokenizer, device, model_name, sent_lst, dataset_name):
    model.eval()
    #softmax_pred_lst = []
    output_file = 'train_softmax_' + model_name + '.txt'
    output_file_path = os.path.join('resources', dataset_name, output_file)
    with open(output_file_path, 'w') as fh:
        for idx, batch in enumerate(dataloader):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)

            logits = outputs.logits

            softmax_logits = F.softmax(logits, dim=2)

            if (idx+1) % 50 == 0:
                print(f'Done {idx+1} steps')
            k = 0
            for i in range(ids.shape[0]):
                tmp_sent_softmax = []
                tmp_train_tokens = tokenizer.convert_ids_to_tokens(ids[i])

                for index, tok in enumerate(tmp_train_tokens):
                    if tok in ['[CLS]', '[SEP]', '[PAD]']:
                        continue 
                    else:
                        tmp_sent_softmax.append(softmax_logits[i][index])
                j = 0
                parts = sent_lst[idx*ids.shape[0] + k].split(' ')
                for tok in parts:
                    if j < len(tmp_sent_softmax):
                        sub_words = tokenizer.tokenize(tok)
                        fh.write(f'{tmp_sent_softmax[j][0]},{tmp_sent_softmax[j][1]},{tmp_sent_softmax[j][2]}\n')
                        j += len(sub_words)
                    else:
                        fh.write(f'0,0,1.0\n')
                k += 1
                fh.write(f'\n')
                #softmax_pred_lst.append(tmp_sent_softmax)

    # return softmax_pred_lst

# def write_softmax_output_to_file(model_softmax_outputs, model_name, dataset_name, tokens, tokenizer):
#     output_file = 'train_softmax_' + model_name + '.txt'
#     output_file_path = os.path.join('resources', dataset_name, output_file)
#     sent_softmax = model_softmax_outputs[0]
#     i = 1 # iterator for softmax outputs
#     j = 0 # iterator for a single sentence softmax outputs
#     with open(output_file_path, 'w') as fh:
#         for tok in tokens:
#             if isinstance(tok, float):
#                 fh.write('\n')
#                 if i >= len(model_softmax_outputs):
#                     break 
#                 sent_softmax = model_softmax_outputs[i]
#                 i += 1
#                 j = 0
#             elif j < len(sent_softmax):
#                 sub_words = tokenizer.tokenize(tok)
#                 fh.write(f'{sent_softmax[j][0]},{sent_softmax[j][1]},{sent_softmax[j][2]}\n')
#                 j += len(sub_words)
#             else:
#                 fh.write(f'0,0,1.0\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--output_file_name', type=str, required=True)

    args = parser.parse_args()

    # read data
    train_data = read_data(args.dataset_name)
    
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    num_labels = len(id2label)

    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    device = 'cuda' if cuda.is_available() else 'cpu'

    #load tokenizer
    tokenizer_dir = os.path.join('resources', args.dataset_name, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # load model
    model_dir = os.path.join('resources', args.dataset_name, args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # loading model to device
    print(device)
    model.to(device)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': False                    }
    train_df = {'Sentence':train_sent, 'Labels':train_label}
    train_df = pd.DataFrame(train_df)
    train_dataset = dataset(train_df, tokenizer, MAX_LEN, label2id, id2label)
    train_dataloader = DataLoader(train_dataset, **train_params)

    get_softmax_output(model, train_dataloader, tokenizer, device, args.model_name, train_sent, args.dataset_name)

    # write_softmax_output_to_file(model_softmax_outputs, args.model_name, args.dataset_name, train_data['Tokens'].tolist(), tokenizer)

if __name__ == '__main__':
    main()
