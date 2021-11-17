from itertools import combinations 
from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd
import os

keyword_masked_filepath = '/media/dmlab/My Passport/DATA/cross-domain/MDSD_masked.json'
save_dir = '/media/dmlab/My Passport/DATA/cross-domain/post-train'
if not os.path.exists(save_dir): os.makedirs(save_dir)
    
def main(docs, save_filepath):    
    with open(save_filepath, 'w') as output_file:
        for doc_idx, doc in tqdm(enumerate(docs)):
            output_file.write('{}\n\n'.format(doc))
        output_file.write('[EOD]')
    print(f'Created {save_filepath}')
        
if __name__ == '__main__':
    all_df = pd.read_json(keyword_masked_filepath)[['domain', 'masked_text']]
    all_df = shuffle(all_df)
    domains = all_df.domain.unique()
    
    for (domain1, domain2) in list(combinations(domains, 2)):
        df = all_df[all_df['domain'].isin([domain1, domain2])]
        docs = df['masked_text'].values
        
        print('Creating dataset for {}&{}..'.format(domain1, domain2))
        save_filepath = os.path.join(save_dir, 'MDSD_{}_for_post.txt'.format('&'.join([domain1, domain2])))
        main(docs, save_filepath)