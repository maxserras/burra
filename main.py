"""Launches different scripts depending on the given command"""
import sys
from src import logger

if __name__ == '__main__':
    # If the command is "train-unprocessable-setfit" train it
    if sys.argv[1] == "train-unprocessable-setfit":
        logger.info("Training unprocessable samples setfit model")
        from src.processors.setfit_algorithm import train_unprocessable_samples_setfit
        train_unprocessable_samples_setfit(test_mode=False)

    elif sys.argv[1] == "train-predict-setfit":
        logger.info("Training unprocessable samples setfit model")
        from src.processors.setfit_algorithm import train_unprocessable_samples_setfit
        train_unprocessable_samples_setfit(test_mode=False)
        from src.processors.setfit_algorithm import predict_with_model
        predict_with_model(test_mode=False, model_name="mserras/setfit-alpaca-es-unprocessable-sample-detection")

    # If the command is "predict-unprocessable-setfit" predict it
    elif sys.argv[1] == "predict-unprocessable-setfit":
        from src.processors.setfit_algorithm import predict_with_model
        predict_with_model(test_mode=False, model_name="mserras/setfit-alpaca-es-unprocessable-sample-detection")
    elif sys.argv[1] == "save":
        from src.etl.sync_metadata import save_progress
        save_progress()
    # Alignment between EN and ES datasets
    elif sys.argv[1] == "align":
        from src.etl.load_alpaca_en_original import align_datasets
        align_datasets()

    elif sys.argv[1] == "enrich":
        from src.processors.enrich_metadata import enrich_metadata_with_model
        enrich_metadata_with_model()
