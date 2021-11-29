from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import pipeline
import numpy as np

def load_tokenizer_and_model(model_name_or_path, num_classes=None, mode='classification'):
    if num_classes is not None: # train
        config = AutoConfig.from_pretrained(model_name_or_path, num_classes=num_classes)
    else: # test
        config = AutoConfig.from_pretrained(model_name_or_path)      
    
    print('Loading tokenizer & model for {}..\n'.format(model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if mode == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    elif mode == 'masking':
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        
    return tokenizer, model

class FeatureExtractor:
    def __init__(self, pretrained_model_name):
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.nlp = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer)
    
    def get_feature(self, text):
        try:
            feature = self.nlp(text)
        except:
            # TEMP: tokenizer(text, truncation=True, padding='max_length', max_length=512).items() 한 다음에 
            # 다시 raw text로 되돌려서 512 tokens 만큼의 text를 사용하는 게 가장 정석적인 방법일 듯
            text = text[:500]   
            
            feature = self.nlp(text)
        feature = np.squeeze(feature)
        return feature.mean(0)