"""Enrich metadata with bias and hate speech detection scores."""
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
from transformers import pipeline

from src import logger, CORPUS_DIR
from src.argilla_client import get_alpaca_es_client, argilla_dataset_generator


def get_bias_classifier():
    tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
    model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")
    bias_classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    return bias_classifier


def get_hate_classifier():
    hate_tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
    hate_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
    hate_classifier = pipeline("text-classification", model=hate_model, tokenizer=hate_tokenizer)
    return hate_classifier


def enrich_metadata_with_model():
    argilla_client = get_alpaca_es_client()
    bias_classifier = get_bias_classifier()
    hate_classifier = get_hate_classifier()
    all_samples = []
    # Load the english alpaca dataset from corpus
    with open(os.path.join(CORPUS_DIR, "alpaca_data_cleaned.json"), "r") as f:
        all_en_samples = json.load(f)
    for sample in tqdm(argilla_dataset_generator(argilla_client, dataset_name="somos-alpaca-es",
                                                 query="NOT _exists_:metadata.bias_score")):
        en_index = sample.metadata.get("en_index")
        logger.debug(f"Processing sample with en_index: {en_index}")
        if not en_index or en_index < 0:
            continue
        en_sample = all_en_samples[en_index]
        sample_text = " ".join(en_sample.values())
        try:
            bias_score = bias_classifier(sample_text)[0]
            sample.metadata["bias_score"] = bias_score
        except:
            logger.warning(f"Error processing sample: {sample_text} for bias detection")
            sample.metadata["bias_score"] = {}

        try:
            hate_score = hate_classifier(sample_text)[0]
            sample.metadata["hate_score"] = hate_score
        except:
            logger.warning(f"Error processing sample: {sample_text} for hate detection")
            sample.metadata["hate_score"] = {}
        all_samples.append(sample)
    argilla_client.log(all_samples, name="somos-alpaca-es", batch_size=250)
