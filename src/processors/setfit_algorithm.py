"""Create a SetFit algorithm to detect errors with image requests and photo addings"""
import json
import os

from tqdm import tqdm
from argilla import TextClassificationRecord

from src import logger, CONFIG, CORPUS_DIR

from src.argilla_client import get_alpaca_es_client, argilla_dataset_generator
from setfit import SetFitTrainer, SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss

from src.queries import merge_queries, TAGGED_ITEMS, AVOID_TRANSLATION_FLAGGED, AVOID_TRANSLATION_ERRORS


def transform(r):
    return {
        "text": f"INSTRUCTION:\n{r['inputs']['1-instruction']}\nINPUT:\n{r['inputs']['2-input']}\nOUTPUT:\n{r['inputs']['3-output']}\n",
        "label": 1 if "UNPROCESSABLE" in r["annotation"] else 0
    }


def instruct_fields_to_text(field_instruction: str, field_input: str, field_output: str):
    """Given the instruction, input and output fields, return a text to be used by setfit"""
    return f"INSTRUCTION:\n{field_instruction}\nINPUT:\n{field_input}\nOUTPUT:\n{field_output}\n"


def sample_to_text(sample: TextClassificationRecord) -> str:
    """Converts and Argilla TextClassificationRecord to a text to be used by setfit"""
    return instruct_fields_to_text(sample.inputs["1-instruction"], sample.inputs["2-input"], sample.inputs["3-output"])


def train_unprocessable_samples_setfit(test_mode: bool = False):
    argilla_client = get_alpaca_es_client()
    unprocessable_query = merge_queries(TAGGED_ITEMS, AVOID_TRANSLATION_FLAGGED, AVOID_TRANSLATION_ERRORS)
    labelled_samples = argilla_client.load("somos-alpaca-es",
                                           query=unprocessable_query,
                                           limit=10_000)
    dataset = labelled_samples.to_datasets()
    dataset = dataset.map(transform)
    setfit_model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    trainer = SetFitTrainer(
        model=setfit_model,
        train_dataset=dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=12,
        num_iterations=16 if not test_mode else 1,
        num_epochs=1,
        column_mapping={"text": "text", "label": "label"},
    )
    # Logging model trained
    logger.info(f"Training setfit model on training dataset...")
    trainer.train()
    logger.warning("Saving locally")
    setfit_model._save_pretrained("backup-model-setfit-unprocessable.pck")
    logger.info(f"Push to huggingface hub")
    trainer.push_to_hub("mserras/setfit-alpaca-es-unprocessable-sample-detection", use_auth_token=CONFIG["HF_TOKEN"])


def predict_with_model(model_name: str = "mserras/setfit-alpaca-es-unprocessable-instructions",
                       metadata_field_name: str = "sf-unprocessable-score",
                       test_mode: bool = False):
    """
    Predict with a setfit model on the alpaca-es dataset and save the predictions as a metadata field

    Parameters
    ----------
    model_name: str - name of the model to use, default is mserras/setfit-alpaca-es-unprocessable-instructions
    metadata_field_name: str - metadata field name to save the predictions
    test_mode: bool - if True, only predict on 10 samples

    Returns
    -------

    """
    logger.info(f"Predicting with model {model_name} on alpaca-es dataset")
    setfit_model = SetFitModel.from_pretrained(model_name)
    logger.info("Model loaded")
    argilla_client = get_alpaca_es_client()
    # Log the predictions in a prediction json
    predicted_samples = []
    identifier_to_score_map = {}
    for sample in tqdm(argilla_dataset_generator(argilla_client, dataset_name="somos-alpaca-es", query=None)):
        text = sample_to_text(sample)
        score = setfit_model.predict_proba([text])[0].tolist()[1]
        identifier_to_score_map[sample.id] = score
        sample.metadata[metadata_field_name] = score
        # Cast tensor to list
        if test_mode:
            logger.info(f"Predicted score: {score} for sample {sample.id} with text {text}")
            if len(predicted_samples) > 10:
                break
        predicted_samples.append(sample)
        if len(predicted_samples) > 250:
            argilla_client.log(predicted_samples, "somos-alpaca-es")
            predicted_samples = []

    if predicted_samples and not test_mode:
        argilla_client.log(predicted_samples, "somos-alpaca-es")

    # Save the predictions in a json file
    with open(os.path.join(CORPUS_DIR, "setfit_unprocessable_scores.json"), "w") as f:
        json.dump(identifier_to_score_map, f)

