# nephased

[[Live Demo]](https://demo-for-nephased.vercel.app/) | [[Prototyping]](https://github.com/angeltamang123/Nepali-Hate-Sentiment-Detection)

> [!Warning]
> This section contains vulgar words

`Nephased` provides a BERT-based classification pipeline
for detecting Nepali text sentiment

## Installation

From TestPyPI:

```bash
pip install nephased[ml]
```

Or you can use the Nephased(finetune of distilbert-base-nepali) from [huggingface](https://huggingface.co/Vyke2000/Nephased)

## Usage

Import `Nephased` module using the following command.

```python
from nephased import Nephased
```

Initialize Nephased

```python
clf = Nephased()
```

- You can pass a single string:

```python
>>> clf.predict("थुक्क पैसा मा बिकने हीजडा")
'PROFANITY_0'
```

- or, a list of string:

```python
>>> clf.predict(["राडिको छोरोको शासन धेर दिन टिक्दैन |", "सुरु मा चाहिँ तैले यो देश छोडनु पर्यो |", "एसको घरमा आगो लाहिदे ।"])
['PROFANITY_1', 'GENERAL', 'VIOLENCE']
```

## About Output

Nephased can distinguish between 4 categories:

- GENERAL : Instance without any profanity or violence.
- PROFANITY_0 : Instance including rude, bad or slander which are not very harsh but offensive words used on day-to-day lives in Nepal.
- PROFANITY_1 : Instance including swear or curse words which are very harsh
- VIOLENCE : Instance including physical assualt or rape and pyromaniac act.

The guidelines for segragating such sentiments are on [NepsaGuidelines](https://github.com/oya163/nepali-sentiment-analysis/blob/master/guidelines/NepsaGuidelines_2020.pdf)

> [!NOTE]
> Nephased is trained on [NepSa](https://github.com/oya163/nepali-sentiment-analysis/blob/master/data/nepcls/csv/ss_ac_at_txt_unbal.csv) dataset \
> By default Nephased preprocesses the input:
>
> - stemming using [nepali-stemmer](https://github.com/oya163/nepali-stemmer)
> - lowering case, punctuation and stopwords removal \
>   you can choose to not preprocess text when initializing Nephased
>
> ```python
> clf = nephased(preprocess_text = False)
> ```
