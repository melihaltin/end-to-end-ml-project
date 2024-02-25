from ml_project import logger
from ml_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ml_project.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from ml_project.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from ml_project.pipeline.stage_04_model_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion"
try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = DataIngestionTrainingPipeline()
        logger.info(f"{STAGE_NAME} completed")
        
except Exception as e:
    logger.error(f"Failed to execute Data Ingestion stage: {str(e)}")
    raise e 



STAGE_NAME = "Data Validation"

try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = DataValidationTrainingPipeline()
        logger.info(f"{STAGE_NAME} completed")
except Exception as e:
        logger.error(f"Failed to execute Data Validation stage: {str(e)}")
        raise e        
        
        
        
        
STAGE_NAME = "Data Transformation"

try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = DataTransformationTrainingPipeline()
        logger.info(f"{STAGE_NAME} completed")
except Exception as e:
        logger.error(f"Failed to execute Data Transformation stage: {str(e)}")
        raise e         


STAGE_NAME = 'Model Training'

try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = ModelTrainingPipeline()
        logger.info(f"{STAGE_NAME} completed")

except Exception as e:
        logger.error(f"Failed to execute Model Training stage: {str(e)}")
        raise e        
        