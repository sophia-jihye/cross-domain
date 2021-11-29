from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy
import numpy as np

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification, reports

post_trained_dirs = sorted([d for d in glob('/data/jihye_data/cross-domain/post-train/*&*_MLM') if os.path.isdir(d)])
post_filepath_format = '/data/jihye_data/cross-domain/data/MDSD_source={}_target={}_DDT_for_post.json'
post_parent_save_dir = '/data/jihye_data/cross-domain/post-train'

def prepare_data(source_domain, target_domain):
    post_filepath = post_filepath_format.format(source_domain, target_domain)
    post_df = pd.read_json(post_filepath)[['text', 'label']]
    
    # relabel
    label_names = sorted(post_df['label'].unique())
    label_to_idx = {label_names[idx]:idx for idx in range(len(label_names))}
    post_df['label'] = post_df['label'].apply(lambda x: label_to_idx[x])

    train_df, val_df = train_test_split(post_df, test_size=.2, shuffle=True, \
        random_state=np.random.randint(1, 100), stratify=post_df['label'].values)
    train_texts, val_texts = train_df['text'].values, val_df['text'].values
    train_labels, val_labels = train_df['label'].values, val_df['label'].values
    return train_texts, train_labels, val_texts, val_labels

def start_post_train(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)
    
if __name__ == '__main__':    
    num_classes = 2
    mode = 'MLM+DDT'
    
    for model_name_or_dir in post_trained_dirs:
        post_domains, post_option = os.path.basename(model_name_or_dir).split('_')
        domain1, domain2 = post_domains.split('&')
        
        # domain1 -> domain2
        source_domain, target_domain = domain1, domain2
        
        train_texts, train_labels, val_texts, val_labels = prepare_data(source_domain, target_domain)
        save_dir = os.path.join(post_parent_save_dir, \
                                'source={}_target={}_{}'.format(source_domain, target_domain, mode))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        start_post_train(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)
        
        # domain2 -> domain1
        source_domain, target_domain = domain2, domain1
        
        train_texts, train_labels, val_texts, val_labels = prepare_data(source_domain, target_domain)
        save_dir = os.path.join(post_parent_save_dir, \
                                'source={}_target={}_{}'.format(source_domain, target_domain, mode))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        start_post_train(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)