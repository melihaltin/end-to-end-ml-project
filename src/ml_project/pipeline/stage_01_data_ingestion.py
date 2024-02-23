from ml_project.config.configuration import ConfigurationManager
from ml_project.entity.config_entity import DataIngestionConfig
from ml_project.components.data_ingestion import DataIngestion
from ml_project import logger


STAGE_NAME = "Data Ingestion"

class DataIngestionStage:
    def __init__(self, config):
        self.config = config
        self.stage_name = STAGE_NAME
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(self.data_ingestion_config)
    
    def execute(self):
        logger.info(f"Starting {self.stage_name} stage")
        self.data_ingestion.download_file()
        self.data_ingestion.extract_zip_file()
        logger.info(f"Completed {self.stage_name} stage")
        return self.data_ingestion_config.unzip_dir
    
    
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file() 
        
        

if __name__ == "__main__":
    try:
        logger.info("Starting Data Ingestion stage")
        obj = DataIngestionTrainingPipeline()
        logger.info("Data Ingestion stage completed")
        
    except Exception as e:
        logger.error(f"Failed to execute Data Ingestion stage: {str(e)}")
        raise e         