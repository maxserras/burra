"""Create a SetFit algorithm to detect errors with image requests and photo addings"""
import json

from tqdm import tqdm

from src import logger

from src.argilla_client import get_alpaca_es_client, argilla_dataset_generator
from setfit import SetFitTrainer, SetFitModel
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss


def transform(r):
	return {
		"text": f"INSTRUCTION:\n{r['inputs']['1-instruction']}\nINPUT:\n{r['inputs']['2-input']}\nOUTPUT:\n{r['inputs']['3-output']}\n",
		"label": 1 if "IMAGE-ERROR" in r["annotation"] else 0
	}


def train_predict_image_fixing_setfit(test_mode: bool = False):
	argilla_client = get_alpaca_es_client()
	labelled_samples = argilla_client.load("somos-alpaca-es", query="status:Validated", limit=10_000)
	dataset = labelled_samples.to_datasets()
	dataset = dataset.map(transform)
	setfit_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
	trainer = SetFitTrainer(
		model=setfit_model,
		train_dataset=dataset,
		loss_class=CosineSimilarityLoss,
		batch_size=4,
		num_iterations=10 if not test_mode else 1,
		num_epochs=1,
		column_mapping={"text": "text", "label": "label"},
	)
	# Logging model trained
	logger.info(f"Training setfit model on training dataset...")
	trainer.train()
	# Log the predictions in a prediction json
	prediction_dict = {}
	for sample in tqdm(argilla_dataset_generator(argilla_client, dataset_name="somos-alpaca-es", query=None)):
		text = f"INSTRUCTION:\n{sample.inputs['1-instruction']}\nINPUT:\n{sample.inputs['2-input']}\nOUTPUT:\n{sample.inputs['3-output']}\n"
		prediction_dict[sample.id] = int(setfit_model.predict([text])[0].item())
		# Cast tensor to int
		if test_mode and len(prediction_dict) > 10:
			break
	# save the json file to "setfit_image_fixing.json" file
	logger.info(f"Saving setfit predictions to setfit_image_fixing.json")
	with open("setfit_image_fixing.json", "w") as f:
		json.dump(prediction_dict, f)

