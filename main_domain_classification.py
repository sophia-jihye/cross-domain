from sklearn.model_selection import train_test_split
from itertools import combinations 
import pandas as pd
import os, copy
import numpy as np

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset
import finetuning_classification

mdsd_unlabeled_filepath = '/data/jihye_data/cross-domain/data/MDSD_unlabeled.json'
finetune_parent_save_dir = '/data/jihye_data/cross-domain/domain-cls'

def start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)
    
if __name__ == '__main__':        

    num_classes = 2
    unlabeled_df = pd.read_json(mdsd_unlabeled_filepath)
    print('Loaded {}'.format(mdsd_unlabeled_filepath))
    model_name_or_dir = 'bert-base-uncased'
    
    ##################################
    ###### Domain classification #####
    ##################################    
    domains = unlabeled_df.domain.unique()
    for (domain1, domain2) in list(combinations(domains, 2)):
        df = copy.copy(unlabeled_df[unlabeled_df['domain'].isin([domain1, domain2])])
        
        relabel_dict = {val:i for i, val in enumerate(sorted(df['domain'].unique()))}
        df['domain'] = df['domain'].apply(lambda x: relabel_dict[x])

        save_dir = os.path.join(finetune_parent_save_dir, '{}'.format('&'.join(sorted([domain1, domain2]))))
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        train_df, val_df = train_test_split(df, test_size=.2, shuffle=True, random_state=np.random.randint(1, 100), stratify=df['domain'].values)
        train_texts, val_texts = train_df['text'].values, val_df['text'].values
        train_labels, val_labels = train_df['domain'].values, val_df['domain'].values
        start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)
            