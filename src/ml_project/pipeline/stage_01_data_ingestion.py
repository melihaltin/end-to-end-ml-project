from ml_project.config.configuration import ConfigurationManager
from ml_project.components.data_ingestion import DataIngestion
from ml_project import logger



STAGE_NAME = "Data Ingestion"

    
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main():
        config = ConfigurationManager()
       
        data_ingestion_config = config.get_data_ingestion_config()
        print(data_ingestion_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file() 
    
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")   
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
        
        
# if __name__ == '__main__':
    # try:
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     obj = DataIngestionTrainingPipeline()
    #     obj.main()
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e