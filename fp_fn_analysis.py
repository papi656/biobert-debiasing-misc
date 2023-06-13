import os
import math
import argparse
import textPreprocessing   

def get_sent_lst(dataset_name):
    file_path = os.path.join('datasets', dataset_name, 'test.txt')

    # getting list of tokens from test.txt file
    test_token_lst = []
    with open(file_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                test_token_lst.append(math.nan)
                continue
            parts = line.split('\t')
            test_token_lst.append(parts[0].strip())

    # converting list of tokens to sentences
    sent_lst = []
    sent = ""
    for tok in test_token_lst:
        if isinstance(tok, float):
            sent = sent[1:]
            sent_lst.append(sent)
            sent = ""
        else:
            sent = sent + ' ' + str(tok)

    if len(sent.strip()) > 0:
        sent_lst.append(sent)

    return sent_lst

    

def get_mention_lst(file_path):
    mention_pos_lst = []
    with open(file_path, 'r') as fh:
        i, pos = 0, 0
        sent_pos = 0
        mention = ""
        for line in fh:
            if len(line.strip()) == 0:
                if len(mention.strip()) > 0:
                    mention_pos_lst.append([mention, pos, sent_pos])
                    mention = ""
                sent_pos += 1
                i += 1
                continue 
            parts = line.split('\t')
            if parts[1].strip() == 'B':
                if len(mention.strip()) > 0:
                    mention_pos_lst.append([mention, pos, sent_pos])
                mention = ""
                mention = parts[0]
                pos = i 
            elif parts[1].strip() == 'I':
                mention = mention + ' ' + parts[0]
            else:
                if len(mention.strip()) > 0:
                    mention_pos_lst.append([mention, pos, sent_pos])
                mention = ""
            i += 1

    return mention_pos_lst

            

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)

    args = parser.parse_args()

    sent_lst = get_sent_lst(args.dataset_name)

    # getting mention list from test data
    test_file_path = os.path.join('datasets', args.dataset_name, 'test.txt')
    test_mention_lst = get_mention_lst(test_file_path)

    # getting mention list from prediction file
    pred_file_path = os.path.join('resources', args.dataset_name, args.pred_file)
    model_mention_lst = get_mention_lst(pred_file_path)
    # with open('test_mention.txt','w') as fh:
    #     for t in test_mention_lst:
    #         fh.write(f'{t}\n')
    # with open('model_mention.txt', 'w') as fh:
    #     for t in model_mention_lst:
    #         fh.write(f'{t}\n')
    preprocessor = textPreprocessing.TextPreprocess()

    fp_lst, fn_lst = [], []
    i, j = 0, 0
    while i < len(model_mention_lst) and j < len(test_mention_lst):
        # correctly identified mention
        if preprocessor.run(model_mention_lst[i][0]) == preprocessor.run(test_mention_lst[j][0]):
            i += 1
            j += 1
        else:
            if model_mention_lst[i][1] < test_mention_lst[j][1]:
                fp_lst.append([test_mention_lst[j][0], model_mention_lst[i][0], model_mention_lst[i][2]])
                i += 1
            else:
                fn_lst.append([test_mention_lst[j][0], model_mention_lst[i][0], test_mention_lst[j][2]])
                j += 1

    if i < len(model_mention_lst):
        fp_lst.append(["[No_Original_mention]", model_mention_lst[i][0], model_mention_lst[i][2]])
        i += 1
    if j < len(test_mention_lst):
        fn_lst.append([test_mention_lst[j][0], "[No_model_mention]", test_mention_lst[j][2]])
        j += 1

    
    predFile_name = args.pred_file
    to_append = predFile_name[predFile_name.find('_')+1:]

    fp_file_name = 'fp_' + to_append
    fn_file_name = 'fn_' + to_append
    fp_file_path = os.path.join('resources', args.dataset_name, fp_file_name)
    fn_file_path = os.path.join('resources', args.dataset_name, fn_file_name)

    with open(fp_file_path, 'w') as fh:
        for word in fp_lst:
            fh.write(f'{word[0]} <--> {word[1]} \n')
            fh.write(f'\t-{sent_lst[word[2]]}\n')

    with open(fn_file_path, 'w') as fh:
        for word in fn_lst:
            fh.write(f'{word[0]} <--> {word[1]}\n')
            fh.write(f'\t-{sent_lst[word[2]]}\n')




if __name__ == '__main__':
    main()
