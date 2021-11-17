from glob import glob
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch

from transformers_helper import load_tokenizer_and_model
from LazyLineByLineTextDataset import LazyLineByLineTextDataset
import post_training_mlm

filepaths = glob('/media/dmlab/My Passport/DATA/cross-domain/post-train/MDSD_*_for_post.txt')
parent_save_dir = '/media/dmlab/My Passport/DATA/cross-domain/post-train'

def main(model_name_or_dir, post_filepath, save_dir):
    
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='masking')
    dataset = LazyLineByLineTextDataset(tokenizer=tokenizer, file_path=post_filepath)
    post_training_mlm.train(tokenizer, model, dataset, save_dir)
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU
    print('NOW USING {}'.format(device))
    model_name_or_dir = 'distilbert-base-uncased'
    
    for post_filepath in filepaths:
        save_dir = os.path.join(parent_save_dir, os.path.basename(post_filepath).split('_')[1])
        main(model_name_or_dir, post_filepath, save_dir)