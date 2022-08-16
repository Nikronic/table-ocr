# core
import numpy as np
# ours: data
from ttocr.data import io
from ttocr.data import preprocessors
# ours: detection
from ttocr.detection import detectors
# ours: utils
from ttocr.version import VERSION as TTOCR_VERSION
# devops
import mlflow
# helpers
from pathlib import Path
import logging
import shutil
import sys


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
    MLFLOW_ARTIFACTS_IMAGES_PATH = MLFLOW_ARTIFACTS_PATH / 'images'
    if MLFLOW_ARTIFACTS_PATH.exists():
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
    if not MLFLOW_ARTIFACTS_PATH.exists():
        MLFLOW_ARTIFACTS_PATH.mkdir(parents=True)
        MLFLOW_ARTIFACTS_LOGS_PATH.mkdir(parents=True)
        MLFLOW_ARTIFACTS_CONFIGS_PATH.mkdir(parents=True)
        MLFLOW_ARTIFACTS_IMAGES_PATH.mkdir(parents=True)
    
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
    __libs = ['ttocr']
    for __l in __libs:
        __libs_logger = logging.getLogger(__l)
        __libs_logger.setLevel(VERBOSE)
        __libs_logger.addHandler(logger_handler)
        __libs_logger.addHandler(stdout_stream_handler)
        __libs_logger.addHandler(stderr_stream_handler)

    # log experiment configs
    MLFLOW_EXPERIMENT_NAME = f'Single column table - {TTOCR_VERSION}'
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    MLFLOW_TAGS = {
        'stage': 'dev'  # dev, beta, production
    }
    mlflow.set_tags(MLFLOW_TAGS)

    logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
    logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

    try:
        # read image
        filename = 'sample/orig/05-col-wo-border.png'
        img_reader = io.CV2ImageReader()
        img = img_reader(filename)

        # copy image for visualization (draw on image)
        copy_image = np.copy(img)

        # convert color to gray
        color_converter = preprocessors.CV2ImageColorConverter(
            mode=preprocessors.CV2ImageColorConverterModes.BGR2GRAY
        )
        gray_img = color_converter(image=img)

        # detect canny edges
        canny_detector = detectors.CannyEdgeDetector(
            threshold1=50,
            threshold2=200,
            aperture_size=3,
            L2_gradient=False
        )
        canny_edges = canny_detector(image=gray_img,
                                     plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

        # detect lines
        line_detector = detectors.ProbabilisticHoughLinesDetector(
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            min_line_length=350,
            max_line_gap=18
        )
        lines = line_detector(image=canny_edges)
        
        # define ocr engine
        ocr_engine = detectors.TesseractOCR(
            l='eng+fas',
            dpi=100,
            psm=6,
            oem=3,
        )
        table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
        table_cell_ocr.vertical_lines = lines[0]
        table_cell_ocr.horizontal_lines = lines[1]
        table_cell_ocr(image=gray_img,
                       plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

    except Exception as e:
        logger.exception(e)
        raise e
    
    # cleanup code
    finally:
        # log artifacts (logs, saved files, etc)
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_PATH)
    