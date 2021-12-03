from glob import glob
import os
import pandas as pd

from LazyLineByLineTextDataset import LazyLineByLineTextDataset
from transformers_helper import load_tokenizer_and_model
import post_training_mlm

post_filepaths = glob('/data/jihye_data/cross-domain/data/MDSD_*_*_for_post.txt')
post_parent_save_dir = '/data/jihye_data/cross-domain/post-train'

def start_post_train(model_name_or_dir, post_filepath, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='masking')
    dataset = LazyLineByLineTextDataset(tokenizer=tokenizer, file_path=post_filepath)
    post_training_mlm.train(tokenizer, model, dataset, save_dir)

if __name__ == '__main__':
    
    model_name_or_dir = 'bert-base-uncased'
    for post_filepath in post_filepaths:
        _, domain, mode, _, _ = os.path.basename(post_filepath).split('_')
        save_dir = os.path.join(post_parent_save_dir, '{}_{}'.format(domain, mode))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        start_post_train(model_name_or_dir, post_filepath, save_dir)