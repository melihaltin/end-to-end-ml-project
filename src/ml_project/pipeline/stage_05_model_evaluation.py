from ml_project.config.configuration import ConfigurationManager
from ml_project.components.model_evaluator import ModelEvaluator
from ml_project import logger
from pathlib import Path


STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main():
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluator(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()
        
        
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        
