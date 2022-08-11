import imp
# core
import numpy as np
import cv2
# ours: data
from ttocr.data import io
from ttocr.data import preprocessors
# devops
import mlflow
# helpers
from pathlib import Path
import logging
import shutil
import sys
import os

from ttocr.detection import detectors


if __name__ == '__main__':
    # globals
    SEED = 8569
    VERBOSE = logging.DEBUG
    DEVICE = 'cuda'

    # configure MLFlow tracking remote server
    #  see `mlflow-server.sh` for port and hostname. Since
    #  we are running locally, we can use the default values.
    mlflow.set_tracking_uri('http://localhost:5000')

    # configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(VERBOSE)
    logger_formatter = logging.Formatter(
        "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
    )

    # setup dirs for mlflow artifacts (logs, configs, etc)
    MLFLOW_ARTIFACTS_PATH = Path('artifacts')
    MLFLOW_ARTIFACTS_LOGS_PATH = MLFLOW_ARTIFACTS_PATH / 'logs'
    MLFLOW_ARTIFACTS_CONFIGS_PATH = MLFLOW_ARTIFACTS_PATH / 'configs'
    if not os.path.exists(MLFLOW_ARTIFACTS_PATH):
        os.makedirs(MLFLOW_ARTIFACTS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_LOGS_PATH)
        os.makedirs(MLFLOW_ARTIFACTS_CONFIGS_PATH)
    else:
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
    
    # Set up root logger, and add a file handler to root logger
    logger_handler = logging.FileHandler(filename=MLFLOW_ARTIFACTS_LOGS_PATH / 'main.log',
                                         mode='w')
    stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
    stderr_stream_handler = logging.StreamHandler(stream=sys.stderr)
    logger_handler.setFormatter(logger_formatter)
    stdout_stream_handler.setFormatter(logger_formatter)
    stderr_stream_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)  # type: ignore
    logger.addHandler(stdout_stream_handler)
    logger.addHandler(stderr_stream_handler)

    # set libs to log to our logging config
    __libs = []
    for __l in __libs:
        __libs_logger = logging.getLogger(__l)
        __libs_logger.setLevel(VERBOSE)
        __libs_logger.addHandler(logger_handler)
        __libs_logger.addHandler(stdout_stream_handler)
        __libs_logger.addHandler(stderr_stream_handler)


    try:
        # read image
        filename = 'sample/orig/01-table.png'
        img_reader = io.CV2ImageReader()
        img = img_reader(filename)
        # copy image for visualization (draw on image)
        copy_image = np.copy(img)

        # convert color to gray
        color_converter = preprocessors.CV2ImageColorConverter()
        gray_img = color_converter(image=img,
                                   mode=preprocessors.CV2ImageColorConverterModes.BGR2GRAY)
        # detect canny edges
        canny_detector = detectors.CannyEdgeDetector()
        canny_edges = canny_detector(image=gray_img, threshold1=50, threshold2=200)
        

    except Exception as e:
        logger.exception(e)
        raise e
    
    # cleanup code
    finally:
        # log artifacts (logs, saved files, etc)
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
    