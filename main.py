"""Launches different scripts depending on the given command"""
import sys
from src import logger

if __name__ == '__main__':
    # If the command is "train-unprocessable-setfit" train it
    if sys.argv[1] == "train-unprocessable-setfit":
        logger.info("Training unprocessable samples setfit model")
        from src.processors.setfit_algorithm import train_unprocessable_samples_setfit
        train_unprocessable_samples_setfit(test_mode=False)
    # If the command is "predict-unprocessable-setfit" predict it
    elif sys.argv[1] == "predict-unprocessable-setfit":
        from src.processors.setfit_algorithm import predict_with_model
        import time
        time.sleep(1800)
        predict_with_model(test_mode=False, model_name="mserras/setfit-alpaca-es-unprocessable-sample-detection")