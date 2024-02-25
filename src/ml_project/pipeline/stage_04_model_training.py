from ml_project.config.configuration import ConfigurationManager
from ml_project.components.model_training import ModelTrainer
from ml_project import logger
from pathlib import Path


STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
        
        
    def main():
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training_config = ModelTrainer(config=model_training_config)
        model_training_config.train()
        
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e    