from glob import glob
import pandas as pd
import os

from transformers_helper import load_tokenizer_and_model

test_filepath = '/media/dmlab/My Passport/DATA/cross-domain/MDSD_masked.json'
finetune_unk_dirs = glob('/media/dmlab/My Passport/DATA/cross-domain/finetune/source=*_unk')
finetune_raw_dir_format = '/media/dmlab/My Passport/DATA/cross-domain/finetune/source={}_raw'

def main(model_name_or_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='classification')
    
if __name__ == '__main__':
    relabel_dict = {'negative':0, 'positive':1}
    test_all_df = pd.read_json(test_filepath)
    test_all_df['label'] = test_all_df['label'].apply(lambda x: relabel_dict[x])
    all_domains = test_all_df['domain'].unique()
    
    for finetune_unk_dir in finetune_unk_dirs:
        domain = os.path.basename(filepath).split('_')[0].replace('source=','')
        finetune_raw_dir = finetune_raw_dir_format.format(domain)
        test_domains = [d for d in all_domains if d != domain]
        
        for test_domain in test_domains:
            test_df = test_all_df[test_all_df['domain']==test_domain][['masked_text', 'label']]
            
            main(finetune_unk_dir)
        