# Burra - Somos NLP Hackaton 2023

This repo contains the code for the participation of [@zolastro](https://github.com/zolastro), [@luisetex](https://github.com/luisetex), [@coronadoh](https://github.com/coronadoh) and [@maxserras](https://github.com/maxserras)
for the Somos NLP Hackaton 2023. 

The main objective of this project was to clean and improve the [Somos alpaca es](https://huggingface.co/datasets/somosnlp/somos-clean-alpaca-es) dataset to train Llama in spanish with instructions.

## Contexts
One of the goals of the Somos NLP Hackaton 2023 was to train an Alpaca model in Spanish. To that end,
the organizers translated the original alpaca dataset that can be found [here](https://github.com/tloen/alpaca-lora).

This corpus had [multiple issues](https://github.com/gururise/AlpacaDataCleaned) such as:

- Hallucinations due to using unprocessable inputs (images, urls, etc.)
- Inconsistencies such as empty outputs, etc.
- Wrong answers to math problems (mostly)

Also, almost no evaluation has been done regarding bias/hate speech, etc. over this corpus.

In addition, when translating the corpus to Spanish we found other issues such as:

- Inconsistencies in the translation
- Sentences that were not translated
- Inability of propagating labels from the original EN corpus due to the lack of alignement.
- Mixed sentences

The goal of this project was to clean the dataset and improve it to train a Llama model in Spanish.

## Outcomes
The principal outcomes provided by the Burra team are:

- [SetFit model](https://huggingface.co/mserras/setfit-alpaca-es-unprocessable-sample-detection) to detect unprocessable samples (images, urls, etc.)
  - Same model using the paraphrase-multilingual-mpnet-v2 model, see [Model Card](https://huggingface.co/mserras/setfit-alpaca-es-unprocessable-sample-detection-multi)
- LangID filtering algorithm to detect samples that were not correctly translated to Spanish
- Alpaca EN dataset & Alpaca ES dataset alignment algorithm
- Evaluation of [Bias Detection algorithm](https://huggingface.co/d4data/bias-detection-model) and [Hate Speech Detection algorithm](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain) over the Alpaca EN Dataset. Propagation of this information to the Alpaca ES dataset
- Dataset Available in HuggingFace [space](https://huggingface.co/spaces/mserras/somos-alpaca-es) and [HF Datasets](https://huggingface.co/datasets/mserras/alpaca-es-hackaton)
- This repo containing the experiments and code used.

### Other experiments & Future work
In addition, we've given a try to some other lines of work and research that we would like to share:

1. Using ChatGPT3.5 to identify inconsistent samples:
  - We tried some prompt engineering that can be found at the `src/prompting`folder.
  - Some results are available at [HF Argilla Space](https://huggingface.co/spaces/mserras/somos-alpaca-es) under the name of "somos-alpaca-es-analysis-chatgpt3_5_turbo"

2. First trials of fine-tuning the Bertin LLama: `src/llama/fine_tune_llama.ipynb'
   - Disclaimer: we dedicated almost zero time to this, so we don't guarantee that neither the script nor the results are correct.

Other task remain as future work:
  - Generate more training samples using other open-licensed LLMs
  - Improve the prompting
  - Train self-correction / reflection over the Alpaca ES
  - Train other SetFit model for math problem detection

## How to use this repo

Before running this, export the PYTHONPATH 
    
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src

Installation can be done as always:

    pip install -r requirements.txt

Then, you need to create the config/envs.json with different environment variables, such as:

    - HF_TOKEN: token to access HuggingFace databases / push models / ...
    - HUB_DATASET_NAME: name of the HF database to sync the changes
    - OPENAI_TOKEN: to use the OpenAI Client

Then, all the entrypoints follow the same logic:

    $ python main.py "command"

Where the commands can be:

    - "train-unprocessable-setfit": train the SetFit model to detect unprocessable samples
    - "predict-unprocessable-setfit": predict the unprocessable samples using the SetFit model
    - "train-predict-setfit": train and predict the unprocessable samples using the SetFit model
    - "save": save the progress from the Argilla Space to the HF Dataset
    - "align": align the Alpaca EN dataset with the Alpaca ES dataset
    - "enrich": enrich the Alpaca ES dataset with the results of the Bias Detection and Hate Speech Detection models
