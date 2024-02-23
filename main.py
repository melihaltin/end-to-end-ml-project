from ml_project import logger
from ml_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion"
try:
        logger.info(f"Starting {STAGE_NAME}")
        obj = DataIngestionTrainingPipeline()
        logger.info(f"{STAGE_NAME} completed")
        
except Exception as e:
    logger.error(f"Failed to execute Data Ingestion stage: {str(e)}")
    raise e 