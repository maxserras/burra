import os
from time import time
import openai
from typing import List
from itertools import tee, zip_longest
import numpy as np
import re
from tqdm import tqdm
from src import CONFIG

from loguru import logger # Temporary fix for logging. Need to set envs.json to load the correct config
from src.argilla_client import get_alpaca_es_client, load_dataset_from_argilla

from prompt_config import ROLES


# For testing purposes, simply set this to a small number
NUM_SAMPLES = 1

# The role of model to use for the prompt
ROLE = "guideline"



def split_text(text: str, window_size: int = 2400) -> List[str]:
    """
    Split text into chunks of size window_size, splitting by periods
    Parameters
    ----------
    text: str - text to split
    window_size: int - size of the chunks

    Returns
    -------
    List[str] - list of chunks
    """
    if len(text) <= window_size:
        text_splited = [text]
    else:
        period_list = np.array([m.start() for m in re.finditer(r"\. ", text)])
        idx = [0]
        period_plus_empty = 2
        for period in np.arange(1, len(text) // window_size):
            indices = period_list[period_list // window_size == period]
            if len(indices) > 0:
                # We want to finish setences by period and not join it to the next starting sentence
                idx.append(indices[0] + period_plus_empty)
        start, end = tee(idx)
        next(end)
        text_splited = [text[i:j] for i, j in zip_longest(start, end)]

    return text_splited


def _get_required_environment_variable(name: str) -> str:
    """
    Get an environment variable or raise an exception if it is not defined
    Parameters
    ----------
    name: str - name of the environment variable

    Returns
    -------
    str - value of the environment variable
    """
    val = os.environ.get(name)
    if not val:
        raise Exception(f"Environment variable {name} not defined, exiting...")
    return val


openai.api_key = _get_required_environment_variable("OPENAI_TOKEN")


# TODO: So far this query is very particular to the full Annotation Guideline role. We should make it more general
def build_query(instruction: str, input: str, output: str) -> str:
    """
    Build a query for the ChatGPT model
    Parameters
    ----------
    instruction: str - instruction to give to the model
    input: str - input to give to the model
    output: str - output to give to the model

    Returns
    -------
    str - query to give to the model
    """
    return f"1. Instruction: {instruction}\n2. Input: {input}\n3. Output:{output}"



def query_gpt(query: str, role: str) -> str:
    """
    Query the ChatGPT model
    Parameters
    ----------
    query: str - query to give to the model
    role: str - role to give to the model

    Returns
    -------
    str - response of the model
    """
    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": role,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return result["choices"][0]["message"]["content"]
    except Exception as expt:
        print(f"Error during querying: {expt}")
        return "chatgpt_error"


def process_annotation_metadata() -> None:
    """
    Process the samples of the alpaca-es dataset and add the tag from gpt-3.5-turbo to the metadata

    Parameters
    ----------

    Returns
    -------

    """
    alpaca_client = get_alpaca_es_client()
    alpaca_dataset = load_dataset_from_argilla(
        alpaca_client,
        dataset_name="somos-alpaca-es",
        limit=NUM_SAMPLES
        )
    samples_to_append = []
    amount_of_flags = 0
    for sample in tqdm(alpaca_dataset):
        sample_metadata = {}
        _input = sample.inputs
        _instruction = _input.get('1-instruction')
        _input = _input.get('2-input')
        _output = _input.get('3-output')
        _query = build_query(_instruction, _input, _output)
        gpt_label = query_gpt(_query, ROLE)
        sample_metadata[f"gpt-{ROLE}"] = gpt_label
        if gpt_label == "chatgpt_error":
            sample_metadata[f"gpt-{ROLE}-flag"] = True
            amount_of_flags += 1

        sample.metadata = sample_metadata
        samples_to_append.append(sample)
    print(f"Amount of flags: {amount_of_flags}")
    alpaca_client.log(samples_to_append, "somos-alpaca-es")
    ds = alpaca_client.DatasetForTextClassification(records=samples_to_append).to_datasets()
    ds.push_to_hub(CONFIG["HUB_DATASET_NAME"], token=CONFIG["HF_TOKEN"])


if __name__ == '__main__':
    process_annotation_metadata()