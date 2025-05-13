import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nepali_stemmer.stemmer import NepStemmer
import string
import nltk
from nltk.corpus import stopwords
import os

class NepaliSentimentClassifier:
    def __init__(self, model_name="Vyke2000/Nephased", preprocess_text= True, quantization_technique="optimum", load_in=4, cache_dir=None):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_text = preprocess_text

        # Determine cache directory for Hugging Face models/tokenizers
        hf_cache_dir = cache_dir if cache_dir else os.path.join("/tmp", "huggingface_cache")
        if cache_dir is None and not os.path.exists(hf_cache_dir) : # Added check for None
            os.makedirs(hf_cache_dir, exist_ok=True)
        print(f"Using Hugging Face cache directory: {hf_cache_dir}")

        # Load tokenizer and model 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)

        if quantization_technique == "optimum":
            # .from_pretrained automatically uses optimum based on device
            # for 
            # quantization .i.e. if a supported CPU is the device then optimum is used
            # else, if device is GPU then the model is loaded with full precision
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=hf_cache_dir)
            self.clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        
        # Code below is an example skeleton for bitsandbytes quantization
        # We'll wait for multi-platform version of bitsandbytes which is still in experiment
        # once there's a stable release we could use code below to quantize for CPU device
        # elif quantization_technique == "bitsandbytes":
        #     if load_in == 4:
        #         self.model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_4bit=True)
        #         self.clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        #     elif load_in == 8:
        #         self.model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_8bit=True)
        #         self.clf = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        #     else:
        #         raise ValueError("Please provide proper quantization option to load the model using bitsandbytes: 4 or 8")
        
        else:
            raise ValueError("Please provide proper quantization technique to load the model. \nSupported quantization library: 'optimum'.")
            
        self.label_map = {0: "GENERAL", 1: "PROFANITY_0", 2: "PROFANITY_1", 3: "VIOLENCE"}

        # Nepali Stemmer
        self.stemmer = NepStemmer()
        
        # Ensure stopwords are available
        self._ensure_stopwords()

    def _ensure_stopwords(self):
        # Nepali stopwords
        nltk_data_path = os.path.join("/tmp", "nltk_data")
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path, exist_ok=True)
        
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)

        try:
            self.nepali_stopwords = stopwords.words('nepali')
        except LookupError:
            nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
            self.nepali_stopwords = stopwords.words('nepali')
            
    def _preprocess_text(self, text):
        """Apply stemming, lowercasing, punctuation removal, and stopword removal."""
        text = self.stemmer.stem(text)  # Apply stemming
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = ' '.join([word for word in text.split() if word not in self.nepali_stopwords])  # Remove stopwords
        return text
        
    def predict(self, data):
        """
        High-level API to classify either:
        1) a single text (string), or
        2) a list of text strings

        Returns:
        - If data is a single string -> a single sentiment string
        - If data is a list of strings -> a list of sentiment strings
        """
        if isinstance(data, str):
            # Single text
            if self.preprocess_text: 
                data = self._preprocess_text(data)
                
            result = self.clf(data)[0]  # returns list of length 1
            return self.label_map[int(result["label"].split("_")[1])]

        elif isinstance(data, list):
            # input is list 
            if self.preprocess_text: 
                data = [self._preprocess_text(x) for x in data]

            results = self.clf(data, batch_size=16)
            results = [self.label_map[int(result["label"].split("_")[1])] for result in results] 
            return results
            
        else:
            raise ValueError("Input must be a string or list of strings")