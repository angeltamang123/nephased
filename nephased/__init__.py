from .model import NepaliSentimentClassifier

_nephased_instance = None

def Nephased():
    """
    Load the pipeline once and reuse the instance.
    """
    global _nephased_instance
    if _nephased_instance is None:
        _nephased_instance = NepaliSentimentClassifier()
    return _nephased_instance

__all__ = ["Nephased"]
