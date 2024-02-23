import os 
import sys
import logging

logging_string = "[%(asctime)s: %(levelname)s: %(module)s:  %(message)s]"

log_dir = "logs"
loge_file = os.path.join(log_dir, "runnning.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format=logging_string ,handlers=[logging.FileHandler(loge_file), logging.StreamHandler(sys.stdout)])


logger= logging.getLogger("ml_project")