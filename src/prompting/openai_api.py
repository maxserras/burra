from time import sleep
from typing import Optional

import openai
import argilla as rg

from src import logger
from src.queries import merge_queries, AVOID_TRANSLATION_FLAGGED, AVOID_TRANSLATION_ERRORS

SYSTEM_TEMPLATE = """
Analice el conjunto dado de entrada, instrucción y respuesta respecto a los siguientes puntos:

 1- [Sí/No] Existe algún contenido ofensivo o tendencioso en cualquiera de los campos.
 
 2- [Sí/No] La instrucción o la entrada requiere la lectura de imágenes, gráficos, URL, páginas web, fotografías, Wikis y similares.
 
 3- [Sí/No] La respuesta contiene información sobre eventos históricos, noticias, estudios de mercado, investigaciones, sucesos etc.
 
 4- [Sí/No] La respuesta contiene información factual sobre marcas, productos, servicios, empresas, personas, etc.
 
 5- [Sí/No] Existen incoherencias entre la instrucción, la entrada y la respuesta.
 
 6- [Sí/No] La respuesta predice información del futuro.
 
Si un punto se devuelve un "Sí", explica el razonamiento. En caso que la respuesta sea "No,", devuelve sólo "No".
"""

PROMPT_TEMPLATE = """
"{}"
"""


def instruct_fields_to_text(field_instruction: str, field_input: str, field_output: str):
    return f"Instrucción:\n{field_instruction}\nEntrada:\n{field_input}\nRespuesta:\n{field_output}\n"


def sample_to_text(sample: rg.TextClassificationRecord) -> str:
    return instruct_fields_to_text(sample.inputs["1-instruction"], sample.inputs["2-input"], sample.inputs["3-output"])


class OpenAIClient:

    def __init__(self, chat_model_name: str = "gpt-3.5-turbo", token: Optional[str] = None):
        logger.info(f"Initializing OpenAI client with model: {chat_model_name}")
        self.__chat_model = chat_model_name
        if not token:
            raise ValueError("OpenAI API key not found!")
        openai.api_key = token

    def send_record_to_api(self, sample: rg.TextClassificationRecord) -> rg.Text2TextRecord:
        """
        Send a sample to OpenAI for evaluation
        Parameters
        ----------
        sample: rg.TextClassificationRecord
            The sample to send to OpenAI for evaluation

        Returns
        -------
        """
        user_prompt = PROMPT_TEMPLATE.format(sample_to_text(sample))
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": user_prompt}
        ]
        logger.info(f"Generating chat completion with model: {self.__chat_model}")
        completion = openai.ChatCompletion.create(
            model=self.__chat_model,
            messages=messages,
            temperature=0.0
        )
        response = {
            "output": completion.choices[0].message["content"],
            "input": user_prompt
        }
        sleep(0.5)
        return rg.Text2TextRecord(
            text=response["input"],
            prediction=[response["output"]],
            id=sample.id,
            metadata=sample.metadata
        )


def analyze_samples_with_openai():
    from multiprocessing import Pool
    from src import CONFIG
    from src.argilla_client import get_alpaca_es_client

    client = get_alpaca_es_client()
    unprocessable_query = merge_queries(AVOID_TRANSLATION_FLAGGED, AVOID_TRANSLATION_ERRORS)

    labelled_samples = client.load("somos-alpaca-es",
                                   query=unprocessable_query,
                                   limit=200)

    openai_client = OpenAIClient(token=CONFIG["OPENAI_TOKEN"])
    with Pool(4) as p:
        samples_to_log = p.map(openai_client.send_record_to_api, labelled_samples)

    client.log(samples_to_log, "somos-alpaca-es-analysis-chatgpt3_5_turbo")
