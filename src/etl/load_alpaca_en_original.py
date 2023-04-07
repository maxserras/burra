"""Loads the alpaca EN original dataset"""
import os

import requests
from datasets import load_dataset
from tqdm import tqdm

from src import CORPUS_DIR, logger
from src.argilla_client import get_alpaca_es_client, argilla_dataset_generator

ALPACA_EN_SOURCE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned.json"


def align_datasets():
	"""Aligns the alpaca EN original dataset with the alpaca ES dataset"""
	# Load the alpaca ES dataset
	es_dataset = load_dataset("somosnlp/somos-clean-alpaca-es", split="train")
	# Load a dataset from the given uri .json
	response = requests.get(ALPACA_EN_SOURCE)
	en_dataset = response.json()

	# Load the alignment file from ./corpus/alpaca.alignment
	with open(os.path.join(CORPUS_DIR, "alpaca.candidates.tsv"), "r") as f:
		alignment = f.read().split('\n')
		alignment = [line.split('\t')[1:] for line in alignment if line]

	text_to_id_map = {}
	for spanish_sample in es_dataset:
		text = spanish_sample["inputs"]['1-instruction'] + ' ' + spanish_sample["inputs"]['2-input'] + ' ' + \
		       spanish_sample["inputs"]['3-output']
		text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
		text_to_id_map[text] = spanish_sample['id']

	english_text_to_index_map = {}
	for en_index, english_sample in enumerate(en_dataset):
		text = english_sample['instruction'] + ' ' + english_sample['input'] + ' ' + english_sample['output']
		text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
		english_text_to_index_map[text] = en_index

	alignment_map = {}
	# Find each text in the given dataset
	for spanish_text, english_text in alignment:
		spanish_text_id = text_to_id_map[spanish_text]
		english_text_id = english_text_to_index_map[english_text]
		alignment_map[spanish_text_id] = english_text_id

	argilla_client = get_alpaca_es_client()
	reviewed_samples = []
	for spanish_sample in tqdm(argilla_dataset_generator(argilla_client, dataset_name="somos-alpaca-es",
	                                                     query="NOT _exists_:metadata.en_index")):
		english_index = alignment_map.get(spanish_sample.id, None)
		if english_index is None:
			logger.info(f"Could not find english index for {spanish_sample.id}")
			spanish_sample.metadata["en_index"] = -1
		else:
			english_instruction = en_dataset[english_index]['instruction']
			if english_instruction == spanish_sample.inputs['1-instruction']:
				if spanish_sample.status != "Validated":
					spanish_sample.status = "Validated"
					spanish_sample.annotation = ["BAD INSTRUCTION"]
					spanish_sample.annotation_agent = "same-instruction-auto"
		spanish_sample.metadata["en_index"] = english_index
		reviewed_samples.append(spanish_sample)
		if len(reviewed_samples) > 1000:
			argilla_client.log(reviewed_samples, "somos-alpaca-es", batch_size=250)
			reviewed_samples = []

	if reviewed_samples:
		argilla_client.log(reviewed_samples, "somos-alpaca-es", batch_size=250)