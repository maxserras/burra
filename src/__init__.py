import os
import json

from loguru import logger

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SRC_DIR, "..", "config")
CORPUS_DIR = os.path.join(SRC_DIR, "..", "corpus")

with open(os.path.join(CONFIG_DIR, "envs.json"), "r") as fp:
    CONFIG = json.load(fp)
