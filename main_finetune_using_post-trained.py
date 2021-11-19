import os
from glob import glob
import pandas as pd

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset
import finetuning_classification

post_trained_dirs = [d for d in glob('/media/dmlab/My Passport/DATA/cross-domain/post-train/*') if os.path.isdir(d)]
train_filepaths = glob('/media/dmlab/My Passport/DATA/cross-domain/train&val/*_train.json')
val_filepath_format = '/media/dmlab/My Passport/DATA/cross-domain/train&val/{}_val.json'
parent_save_dir = '/media/dmlab/My Passport/DATA/cross-domain/finetune_using_post-trained'
if not os.path.exists(parent_save_dir): os.makedirs(parent_save_dir)

def main(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)
    
if __name__=='__main__':
#     model_name_or_dir = 'distilbert-base-uncased'
    relabel_dict = {'negative':0, 'positive':1}
    num_classes = 2
    
    for train_filepath in train_filepaths:
        for mode in ['unk', 'raw']:
            domain = os.path.basename(train_filepath).split('_')[0]
            model_name_or_dirs = [d for d in post_trained_dirs if domain in d]
            for model_name_or_dir in model_name_or_dirs:
                save_dir = os.path.join(parent_save_dir, 'source={}_{}_post={}'.format(domain, mode, os.path.basename(model_name_or_dir)))
                val_filepath = val_filepath_format.format(domain)
                train_df = pd.read_json(train_filepath)[['text', 'masked_text', 'label']]
                train_df['label'] = train_df['label'].apply(lambda x: relabel_dict[x])
                val_df = pd.read_json(val_filepath)[['text', 'masked_text', 'label']]
                val_df['label'] = val_df['label'].apply(lambda x: relabel_dict[x])

                if mode == 'unk':
                    train_texts, val_texts = train_df['masked_text'].values, val_df['masked_text'].values
                elif mode == 'raw':
                    train_texts, val_texts = train_df['text'].values, val_df['text'].values

                train_labels, val_labels = train_df['label'].values, val_df['label'].values
                main(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)