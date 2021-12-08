import pandas as pd
import os, torch
from glob import glob
from numpy import dot
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
tqdm.pandas()

from transformers_helper import FeatureExtractor

mdsd_labeled_filepath = '/data/jihye_data/cross-domain/data/MDSD_labeled.json'
mdsd_unlabeled_filepath = '/data/jihye_data/cross-domain/data/MDSD_unlabeled.json'
domain_save_dirs = glob('/data/jihye_data/cross-domain/domain-cls/*')
cos_score_threshold = 0.95 # None

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

def do_experiment(unlabeled_df, source_domain, feature_extractor, domain_save_dir, labeled_df, target_domain):
    
    # Source texts from unlabeled data
    records = []
    for text in tqdm(unlabeled_df[unlabeled_df['domain']==source_domain]['text'].values):
        vec = feature_extractor.get_cls_embedding(text)
        records.append(('unlabeled', text, vec))
    source_df = pd.DataFrame(records, columns=['from', 'text', 'dom-cls'])

    filepath = os.path.join(domain_save_dir, '{}_{}.csv'.format('source', source_domain))
    source_df.to_csv(filepath, index=False)
    print('Created {}'.format(filepath))

    if cos_score_threshold is None: # Get the most similar document
        records = []
        # Target texts from labeled data
        for text in tqdm(labeled_df[labeled_df['domain']==target_domain]['text'].values):
            vec = feature_extractor.get_cls_embedding(text)

            source_df['cos'] = source_df['dom-cls'].apply(lambda x: cos_sim(vec, x))
            most_similar_row = source_df.sort_values(by=['cos'], ascending=False, axis=0).iloc[0]
            most_similar_text, most_similar_score = most_similar_row['text'], most_similar_row['cos']
            source_df.drop(columns=['cos'], inplace=True)

            records.append(('labeled', text, vec, most_similar_text, most_similar_score))

        # Target texts from unlabeled data
        for text in tqdm(unlabeled_df[unlabeled_df['domain']==target_domain]['text'].values):
            vec = feature_extractor.get_cls_embedding(text)

            source_df['cos'] = source_df['dom-cls'].apply(lambda x: cos_sim(vec, x))
            most_similar_row = source_df.sort_values(by=['cos'], ascending=False, axis=0).iloc[0]
            most_similar_text, most_similar_score = most_similar_row['text'], most_similar_row['cos']
            source_df.drop(columns=['cos'], inplace=True)

            records.append(('unlabeled', text, vec, most_similar_text, most_similar_score))        

        target_df = pd.DataFrame(records, columns=['from', 'text', 'dom-cls', 'most-similar_text', 'most-similar_score'])
        filepath = os.path.join(domain_save_dir, '{}_{}.csv'.format('target', target_domain))
        target_df.to_csv(filepath, index=False)
        print('Created {}'.format(filepath))
    
    else:
        # Target texts from labeled data
        records = set()
        for text in tqdm(labeled_df[labeled_df['domain']==target_domain]['text'].values):
            vec = feature_extractor.get_cls_embedding(text)

            source_df['cos'] = source_df['dom-cls'].apply(lambda x: cos_sim(vec, x))
            records.update(source_df[source_df['cos']>=cos_score_threshold]['text'].unique())
            source_df.drop(columns=['cos'], inplace=True)

        target_df = pd.DataFrame(list(records), columns=['most-similar_text'])
        target_df['from'] = 'labeled'
        filepath = os.path.join(domain_save_dir, '{}-{}_{}.csv'.format('target', cos_score_threshold, target_domain))
        target_df.to_csv(filepath, index=False)
        print('Created {}'.format(filepath))        
    
if __name__ == '__main__':        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_df = pd.read_json(mdsd_labeled_filepath)
    print('Loaded {}'.format(mdsd_labeled_filepath))
    unlabeled_df = pd.read_json(mdsd_unlabeled_filepath)
    print('Loaded {}'.format(mdsd_unlabeled_filepath))
    
    for domain_save_dir in domain_save_dirs:
        feature_extractor = FeatureExtractor(domain_save_dir, device)
        domains = os.path.basename(domain_save_dir).split('&')
        
        source_domain, target_domain = domains[0], domains[1]
        do_experiment(unlabeled_df, source_domain, feature_extractor, domain_save_dir, labeled_df, target_domain)
        
        source_domain, target_domain = domains[1], domains[0]
        do_experiment(unlabeled_df, source_domain, feature_extractor, domain_save_dir, labeled_df, target_domain)
            