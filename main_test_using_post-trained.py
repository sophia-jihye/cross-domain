from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy

from transformers_helper import load_tokenizer_and_model
from CustomDataset import encode_for_inference
import finetuning_classification, reports

test_filepath = '/media/dmlab/My Passport/DATA/cross-domain/MDSD_masked.json'
finetune_unk_dirs = glob('/media/dmlab/My Passport/DATA/cross-domain/finetune_using_post-trained/source=*_unk_post=*')
parent_save_dir = '/media/dmlab/My Passport/DATA/cross-domain/test_using_post-trained'

def main(device, model_name_or_dir, df, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, mode='classification')
    model = model.to(device)
    
    print('Inferencing..\n')
    df['predicted_label'] = df['text'].progress_apply(lambda x: finetuning_classification.inference(model, *encode_for_inference(device, tokenizer, x)))
    
    # Save results
    df['correct'] = df.apply(lambda x: x.true_label==x.predicted_label, axis=1)
    labels, preds = df.true_label, df.predicted_label
    accuracy = len(df[df['correct']==True]) / len(df)

    csv_filepath = os.path.join(save_dir, 'results.csv')
    df.to_csv(csv_filepath, index=False)
    
    report_filepath = os.path.join(save_dir, 'classification_report.csv')
    reports.create_classification_report(labels, preds, accuracy, report_filepath)
    
    confusion_filepath = os.path.join(save_dir, 'confusion_matrix.csv')
    reports.create_confusion_matrix(labels, preds, confusion_filepath)
    
if __name__ == '__main__':
    relabel_dict = {'negative':0, 'positive':1}
    test_all_df = pd.read_json(test_filepath)
    test_all_df['label'] = test_all_df['label'].apply(lambda x: relabel_dict[x])
    all_domains = test_all_df['domain'].unique()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for finetune_unk_dir in finetune_unk_dirs:
        domain = os.path.basename(finetune_unk_dir).split('_')[0].replace('source=','')
        test_domain = os.path.basename(finetune_unk_dir).split('_')[2].replace('post=','').replace(domain, '').replace('&', '')
        finetune_raw_dir = finetune_unk_dir.replace('unk', 'raw')
        
        test_df = copy.copy(test_all_df[test_all_df['domain']==test_domain][['masked_text', 'label']])
        test_df.columns = ['text', 'true_label']

        save_dir = os.path.join(parent_save_dir, 'target={}_source={}_{}'.format(test_domain, domain, 'unk'))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        main(device, finetune_unk_dir, test_df, save_dir)

        save_dir = os.path.join(parent_save_dir, 'target={}_source={}_{}'.format(test_domain, domain, 'raw'))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        main(device, finetune_raw_dir, test_df, save_dir)
        