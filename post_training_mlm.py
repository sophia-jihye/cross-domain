from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

def train(tokenizer, model, dataset, save_dir, mlm_prob=0.15):
    model.train()
    
    training_args = TrainingArguments(
        output_dir = save_dir,
        num_train_epochs = 5,
        per_device_train_batch_size = 4,
        save_steps = 10000,
        save_total_limit = 3,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, mlm = True, mlm_probability = mlm_prob
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
    )
    
    print('[Post-training (MLM)] Training..')
    trainer.train()
    
    tokenizer.save_pretrained(save_dir)
    trainer.save_model(save_dir)
    print('[Post-training (MLM)] Saved trained model at {}'.format(save_dir))
