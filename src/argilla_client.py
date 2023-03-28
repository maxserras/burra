from typing import Optional, List

import argilla as rg
from src import logger


def get_alpaca_es_client():
    """Get the alpaca-es database from argilla"""
    import argilla as rg
    rg.init(api_key="team.apikey",
            api_url="https://mserras-somos-alpaca-es.hf.space",
            workspace="team")
    return rg


def get_alpaca_en_client():
    """get the alpaca-en dataset from argilla"""
    import argilla as rg
    rg.init(api_key="team.apikey", api_url="https://mserras-somos-alpaca-en.hf.space", workspace="team")
    return rg


def get_prompt_client():
    # Todo.
    pass


def load_dataset_from_argilla(rg_agent, dataset_name: str,
                              limit: int = -1,
                              query: Optional[str] = None) -> List[rg.TextClassificationRecord]:
    """
    Load a dataset from Argilla. If limit is -1, it will load all the dataset. The search can be filtered using
    an ElasticSearch query

    Parameters
    ----------
    rg_agent: argilla agent to use
    dataset_name: str - name of the dataset to load
    limit: int - number of records to load. If -1, it will load all the dataset
    query: str - ElasticSearch query to filter the results

    Returns
    -------
    List of TextClassificationRecord

    """
    if limit < 0:
        limit = 1000
        records = []
        initial_batch = rg_agent.load(dataset_name, limit=limit, query=query)
        logger.debug(f"[RB] Loaded the initial batch of {limit} from {dataset_name}.")
        while initial_batch:
            records += initial_batch
            logger.debug(f"[RB] Total loaded {len(records)}")
            initial_batch = rg_agent.load(dataset_name, limit=limit, query=query, id_from=initial_batch[-1].id)
            logger.debug(f"[RB] Loaded a batch of {len(records)} from {dataset_name}.")
    else:
        records = rg_agent.load(dataset_name, limit=limit, query=query)
    logger.success(f"[RB]A total of {len(records)} samples loaded from {dataset_name}! Casting them to Sample.")
    return list(records)


if __name__ == '__main__':
    client = get_alpaca_es_client()
    print(client.load("somos-alpaca-es", limit=1)[0])
