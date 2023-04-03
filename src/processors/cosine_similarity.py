from src import CONFIG, logger
from src.argilla_client import get_alpaca_es_client, argilla_dataset_generator
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)


def __cosine_similarity(text1: str, text2: str) -> float:
    """
    Get the cosine similarity between two texts.
    Parameters
    ----------
    text1: str - first text
    text2: str - second text

    Returns
    -------
    float - cosine similarity between the two texts
    """

    embeddings = model.encode([text1, text2])
    cosine_similarity = np.inner(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    
    return cosine_similarity


def get_cosine_similarity(text: str) -> (float):
    """
    Given a sentence from the translated corpus, get the original text from the Alpaca corpus.
    Parameters
    ----------
    sentence: str - sentence to get the original text from

    Returns
    -------
    str - original text of the sentence
    """

    # Remove tabs and return characters from the text:
    text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

    # Search in the candidates file:
    alignment_file = '/data/tmp/burra/corpus/alpaca.candidates.tsv'
    original = None

    with open(alignment_file, 'r') as f:
        for line in f:
            fields = line.split('\t')
            if len(fields) != 3:
                continue
            translated_text = fields[1]
            original_text = fields[2]
            # We also have the alignment score that we can use to filter out bad alignments, but for now we ignore it:
            alignment_score = float(fields[0])

            if text == translated_text:
                original = original_text
                break

    
    if original is None:
        return 0.0
    
    
    # Get the cosine similarity:
    cosine_similarity = __cosine_similarity(text, original)
    return cosine_similarity


    
def process_cosine_similarity_metadata(flag_threshold: str = 0.2) -> None:
    """
    Process the samples of the alpaca-es dataset and add the cosine similarity metadata to the samples.

    Parameters
    ----------
    flag_threshold: float - threshold to consider a sample as a cosine similarity flag

    Returns
    -------

    """
    alpaca_client = get_alpaca_es_client()
    samples_to_append = []
    amount_of_flags = 0
    for sample in tqdm(argilla_dataset_generator(alpaca_client, dataset_name="somos-alpaca-es", query=None)):
        sample_metadata = {}
        # Combine the 1-instruction, 2-input and 3-output fields into a single text:
        text = sample.inputs['1-instruction'] + ' ' + sample.inputs['2-input'] + ' ' + sample.inputs['3-output']
        value = get_cosine_similarity(text)

        # TODO: Add the cosine similarity to the metadata

if __name__ == '__main__':
    process_cosine_similarity_metadata()