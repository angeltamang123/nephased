from .model import NepaliSentimentClassifier

def Nephased(preprocess_text=True, quantization_technique="optimum", load_in=4):
    return NepaliSentimentClassifier(preprocess_text=preprocess_text,
                                     quantization_technique=quantization_technique,
                                     load_in=load_in)

__all__ = ["Nephased"]
