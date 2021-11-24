from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification, reports

post_trained_dirs = [d for d in glob('/data/jihye_data/cross-domain/data/post-train/*&*_*') if os.path.isdir(d)]
mdsd_labeled_filepath = '/data/jihye_data/cross-domain/data/MDSD_labeled.json'
finetune_parent_save_dir = '/data/jihye_data/cross-domain/finetune_{}'

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
    
    
if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict, num_classes = {'negative':0, 'positive':1}, 2
    labeled_df = pd.read_json(mdsd_labeled_filepath)
    labeled_df['label'] = labeled_df['label'].apply(lambda x: relabel_dict[x])
    
    ####################################
    ###### Use post-trained models #####
    ####################################        
    # For each task, we employ a 5-fold cross-validation protocol
    # The results we report are the averaged performance of each model across these 5 folds.
    for finetune_idx in range(5):
        for source_domain in labeled_df['domain'].unique():
            
            model_name_or_dirs = [d for d in post_trained_dirs if source_domain in d]
            for model_name_or_dir in model_name_or_dirs:
                
                post_domain, post_option = os.path.basename(model_name_or_dir).split('_')
                test_domain = post_domain.replace(source_domain, '').replace('&', '')
                
                save_dir = os.path.join(finetune_parent_save_dir.format(finetune_idx), 'source={}_post={}_target={}'.format(source_domain, '-'.join([post_domain, post_option]), test_domain))
                if not os.path.exists(save_dir): os.makedirs(save_dir)

                source_df = labeled_df[labeled_df['domain']==source_domain]
                
                # In each fold, 1600 balanced samples are randomly selected from the labeled data for training and the rest 400 for validation.
                train_df, val_df = train_test_split(source_df, test_size=.2, shuffle=True, stratify=source_df['label'].values)

                # (Raw source data)
                train_texts, val_texts = train_df['text'].values, val_df['text'].values
                train_labels, val_labels = train_df['label'].values, val_df['label'].values
                
                start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)
                
                # Test (Raw target data)
                test_df = copy.copy(labeled_df[labeled_df['domain']==test_domain][['text', 'label']])
                test_df.columns = ['text', 'true_label']
                
                start_test(device, save_dir, test_df, save_dir)
                
    ##########################
    ###### No post-train #####
    ##########################
    model_name_or_dir = 'bert-base-uncased'
    for finetune_idx in range(5):
        for source_domain in labeled_df['domain'].unique():
            for test_domain in [d for d in labeled_df['domain'].unique() if d != source_domain]:
                
                save_dir = os.path.join(finetune_parent_save_dir.format(finetune_idx), 'source={}_post=None_target={}'.format(source_domain, test_domain))
                if not os.path.exists(save_dir): os.makedirs(save_dir)

                source_df = labeled_df[labeled_df['domain']==source_domain]
                
                # In each fold, 1600 balanced samples are randomly selected from the labeled data for training and the rest 400 for validation.
                train_df, val_df = train_test_split(source_df, test_size=.2, shuffle=True, stratify=source_df['label'].values)

                # (Raw source data)
                train_texts, val_texts = train_df['text'].values, val_df['text'].values
                train_labels, val_labels = train_df['label'].values, val_df['label'].values
                
                start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir)
                
                # Test (Raw target data)
                test_df = copy.copy(labeled_df[labeled_df['domain']==test_domain][['text', 'label']])
                test_df.columns = ['text', 'true_label']
                
                start_test(device, save_dir, test_df, save_dir)
            