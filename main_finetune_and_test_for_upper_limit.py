from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification, reports

train_filepaths = glob('/media/dmlab/My Passport/DATA/cross-domain/kfold_train&val/*_train.json')
val_filepath_format = '/media/dmlab/My Passport/DATA/cross-domain/kfold_train&val/{}_k={}_val.json'
test_filepath_format = '/media/dmlab/My Passport/DATA/cross-domain/kfold_train&val/{}_k={}_test.json'
parent_save_dir = '/media/dmlab/My Passport/DATA/cross-domain/finetune_upper_limit'
if not os.path.exists(parent_save_dir): os.makedirs(parent_save_dir)

def main(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir, device, df):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)
    
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
    
if __name__=='__main__':
    model_name_or_dir = 'distilbert-base-uncased'
    relabel_dict = {'negative':0, 'positive':1}
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for train_filepath in train_filepaths:
        for mode in ['unk', 'raw']:
            domain = os.path.basename(train_filepath).split('_')[0]
            kfold_idx = os.path.basename(train_filepath).split('_')[1].replace('k=', '')
            save_dir = os.path.join(parent_save_dir, '{}_{}_k={}'.format(domain, mode, kfold_idx))
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            val_filepath = val_filepath_format.format(domain, kfold_idx)
            train_df = pd.read_json(train_filepath)[['text', 'masked_text', 'label']]
            train_df['label'] = train_df['label'].apply(lambda x: relabel_dict[x])
            val_df = pd.read_json(val_filepath)[['text', 'masked_text', 'label']]
            val_df['label'] = val_df['label'].apply(lambda x: relabel_dict[x])

            if mode == 'unk':
                train_texts, val_texts = train_df['masked_text'].values, val_df['masked_text'].values
            elif mode == 'raw':
                train_texts, val_texts = train_df['text'].values, val_df['text'].values

            train_labels, val_labels = train_df['label'].values, val_df['label'].values
            
            test_filepath = test_filepath_format.format(domain, kfold_idx)
            df = pd.read_json(test_filepath)[['text', 'label']] # Raw texts
            df['label'] = df['label'].apply(lambda x: relabel_dict[x])
            df.columns = ['text', 'true_label']
            main(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir, device, df)