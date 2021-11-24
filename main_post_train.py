from glob import glob
import os
import pandas as pd

from LazyLineByLineTextDataset import LazyLineByLineTextDataset
from transformers_helper import load_tokenizer_and_model
import post_training_mlm

TEMP_ALREADY_COMPLETED_DIRS = ['books&dvd_raw', 'books&electronics_random', 'books&kitchen_raw', 'dvd&electronics_random', 'dvd&electronics_raw', 'dvd&kitchen_keyword', 'electronics&kitchen_raw']

post_filepaths = glob('/data/jihye_data/cross-domain/data/MDSD_*&*_*_for_post.txt')
post_parent_save_dir = '/data/jihye_data/cross-domain/post-train'

post_trained_dirs = [d for d in glob(os.path.join(post_parent_save_dir, '*&*_*')) if os.path.isdir(d)]
mdsd_labeled_filepath = '/media/dmlab/My Passport/DATA/cross-domain/data/MDSD_labeled.json'
finetune_parent_save_dir = '/media/dmlab/My Passport/DATA/cross-domain/finetune_{}'

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
        if save_dir in TEMP_ALREADY_COMPLETED_DIRS: continue   # SKIP completed items.
        start_post_train(model_name_or_dir, post_filepath, save_dir)