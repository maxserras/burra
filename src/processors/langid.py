import string

from ftlangdetect import detect
from tqdm import tqdm

from src.argilla_client import get_alpaca_es_client, load_dataset_from_argilla


def __remove_numbers_from_string(text: str) -> str:
    """
    Remove numbers from a string
    Parameters
    ----------
    text: str - text to remove numbers from

    Returns
    -------
    str - text without numbers
    """
    return "".join([c for c in text if not c.isdigit()])


def __remove_punctuation_from_string(text: str) -> str:
    """
    Remove punctuation from a string
    Parameters
    ----------
    text: str - text to remove punctuation from

    Returns
    -------
    str - text without punctuation
    """
    return "".join([c for c in text if c not in string.punctuation])


def _is_text_exception(text: str, min_tokens: int = 4) -> bool:
    """
    Checks if the current text can be identified as some exception, such as it being a set of numbers,
    really short text, a piece of code etc.

    Parameters
    ----------
    text: str - text to check
    min_tokens: int - minimum number of tokens to consider the text valid

    Returns
    -------
    bool - True if the text is an exception, False otherwise
    """
    is_code = "=" in text
    cleaned_text = __remove_numbers_from_string(text.strip())
    cleaned_text = __remove_punctuation_from_string(cleaned_text)
    is_short = len(cleaned_text.split()) < min_tokens
    return not (is_code or is_short)


def process_langid_metadata(flag_threshold: str = 0.5) -> None:
    """
    Process the samples of the alpaca-es dataset and add the langid scores to the metadata

    Parameters
    ----------
    flag_threshold: float - threshold to consider a sample as a translation flag

    Returns
    -------

    """
    alpaca_client = get_alpaca_es_client()
    alpaca_dataset = load_dataset_from_argilla(alpaca_client, dataset_name="somos-alpaca-es")
    samples_to_append = []
    amount_of_flags = 0
    for sample in tqdm(alpaca_dataset):
        sample_metadata = {}
        for key, value in sample.inputs.items():
            if not _is_text_exception(value):
                detect_result = detect(text=value.replace('\n', " "))
                sample_metadata[f"{key}-{detect_result['lang']}"] = detect_result["score"]
                if detect_result["lang"] != "es" and detect_result["score"] > flag_threshold:
                    sample_metadata[f"translation-flag-{key}"] = True
                    amount_of_flags += 1

        sample.metadata = sample_metadata
        samples_to_append.append(sample)
    print(f"Amount of flags: {amount_of_flags}")
    alpaca_client.log(samples_to_append, "somos-alpaca-es")
    # Todo: Push to hub


if __name__ == '__main__':
    process_langid_metadata()