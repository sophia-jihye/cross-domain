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

mdsd_labeled_filepath = '/data/jihye_data/cross-domain/data/MDSD_labeled.json'
finetune_parent_save_dir = '/data/jihye_data/cross-domain/finetune_{}'
kfold_num = 1

def start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)

def start_test(device, model_name_or_dir, df, save_dir):
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
    
def do_experiment(device, save_dir, model_name_or_dir, num_classes, labeled_df, source_domain, test_domain):
    source_df = labeled_df[labeled_df['domain']==source_domain]
    train_df, val_df = train_test_split(source_df, test_size=.2, shuffle=True, random_state=np.random.randint(1, 100), stratify=source_df['label'].values)
    train_texts, val_texts = train_df['text'].values, val_df['text'].values
    train_labels, val_labels = train_df['label'].values, val_df['label'].values
    start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)

    test_df = copy.copy(labeled_df[labeled_df['domain']==test_domain][['text', 'label']])
    test_df.columns = ['text', 'true_label']
    start_test(device, save_dir, test_df, save_dir)
    
if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict, num_classes = {'negative':0, 'positive':1}, 2
    labeled_df = pd.read_json(mdsd_labeled_filepath)
    print('Loaded {}'.format(mdsd_labeled_filepath))
    labeled_df['label'] = labeled_df['label'].apply(lambda x: relabel_dict[x])
    
    ###### Use post-trained models #####    
    for finetune_idx in range(kfold_num):
        for test_domain in labeled_df['domain'].unique():            
            
            #####################################
            ###### Source+Target domain MLM #####
            #####################################    
            post_trained_dirs = sorted([d for d in glob('/data/jihye_data/cross-domain/post-train/*&*_ST') if os.path.isdir(d)])
            model_name_or_dirs = [d for d in post_trained_dirs if test_domain in d]
            for model_name_or_dir in model_name_or_dirs:
                post_domain, post_mode = os.path.basename(model_name_or_dir).split('_')
                source_domain = post_domain.replace(test_domain, '').replace('&', '')
                save_dir = os.path.join(finetune_parent_save_dir.format(finetune_idx), 'source={}_post={}_target={}'.format(source_domain, post_mode, test_domain))
                if not os.path.exists(save_dir): os.makedirs(save_dir)

                do_experiment(device, save_dir, model_name_or_dir, num_classes, labeled_df, source_domain, test_domain)
            
            ##############################
            ###### Target domain MLM #####
            ##############################    
            post_trained_dirs = sorted([d for d in glob('/data/jihye_data/cross-domain/post-train/*_T') if os.path.isdir(d)])
            model_name_or_dirs = [d for d in post_trained_dirs if test_domain in d]
            for model_name_or_dir in model_name_or_dirs:
                post_domain, post_mode = os.path.basename(model_name_or_dir).split('_')                
                for source_domain in [d for d in labeled_df['domain'].unique() if d!=test_domain]:
                    save_dir = os.path.join(finetune_parent_save_dir.format(finetune_idx), 'source={}_post={}_target={}'.format(source_domain, post_mode, test_domain))
                    if not os.path.exists(save_dir): os.makedirs(save_dir)

                    do_experiment(device, save_dir, model_name_or_dir, num_classes, labeled_df, source_domain, test_domain)
                
            ##########################
            ###### No post-train #####
            ##########################                
            model_name_or_dir = 'bert-base-uncased'
            for source_domain in [d for d in labeled_df['domain'].unique() if d != test_domain]:
                save_dir = os.path.join(finetune_parent_save_dir.format(finetune_idx), 'source={}_post=None_target={}'.format(source_domain, test_domain))
                if not os.path.exists(save_dir): os.makedirs(save_dir)

                do_experiment(device, save_dir, model_name_or_dir, num_classes, labeled_df, source_domain, test_domain)
            