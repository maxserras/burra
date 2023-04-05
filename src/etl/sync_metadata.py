import argilla as rg

import pandas as pd
import requests
from datasets import load_dataset, concatenate_datasets

from src import logger, CONFIG
from src.argilla_client import get_alpaca_es_client


def sync_metadata_info():
	"""Sync metadata info from our argilla dataset to the challenge dataset"""
	argilla_client = get_alpaca_es_client()
	samples_with_metadata = argilla_client.load("somos-alpaca-es", query="metadata:*", limit=10_000)
	logger.info(f"Loaded {len(samples_with_metadata)} samples with metadata")
	# Get the sample identifiers and metadata in a map
	samples_with_metadata = {s.id: s.metadata for s in samples_with_metadata}
	# Load these samples from the challenge dataset
	dataset = load_dataset("somosnlp/somos-clean-alpaca-es", split="train")
	dataset = dataset.remove_columns("metrics")
	records = rg.DatasetForTextClassification.from_datasets(dataset)
	# Write the metadata to the challenge dataset for the identified samples
	for i, sample in enumerate(records):
		if sample.id in samples_with_metadata:
			logger.info(f"Found metadata for sample {sample.id}")
			sample.metadata = samples_with_metadata[sample.id]

	# Convert back to dataset
	dataset = records.to_datasets()
	# Push the dataset to the hub
	dataset.push_to_hub("DATASET_NAME_PLACEHOLDER", token=CONFIG["HF_TOKEN"])
