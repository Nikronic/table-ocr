# core
import numpy as np
# ours: data
from ttocr.data import io
from ttocr.data import preprocessors
# ours: detection
from ttocr.detection import DetectionMode
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
    MLFLOW_EXPERIMENT_NAME = f'Fix#5 - {TTOCR_VERSION}'
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    MLFLOW_TAGS = {
        'stage': 'dev'  # dev, beta, production
    }
    mlflow.set_tags(MLFLOW_TAGS)

    logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
    logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

    try:
        # choose table detection method
        DETECTION_MODE = DetectionMode.ML_SINGLE_COLUMN_TABLE
        logger.info(f'Detection mode: {DETECTION_MODE}')

        # read image
        filename = 'sample/orig/03-col-with-border.png'
        img_reader = io.CV2ImageReader()
        img = img_reader(filename)

        # copy image for visualization (draw on image)
        copy_image = np.copy(img)

        # convert color to gray
        color_converter = preprocessors.CV2ImageColorConverter(
            mode=preprocessors.CV2ImageColorConverterModes.BGR2GRAY
        )
        img = color_converter(image=img)

        if DETECTION_MODE == DetectionMode.ML_FULL_TABLE:
            # detect canny edges
            canny_detector = detectors.CannyEdgeDetector(
                threshold1=50,
                threshold2=200,
                aperture_size=3,
                L2_gradient=False
            )
            canny_edges = canny_detector(image=img,
                                        plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

            # detect lines
            line_detector = detectors.ProbabilisticHoughLinesDetector(
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                min_line_length=50,
                max_line_gap=10
            )
            lines = line_detector(image=canny_edges)
            
            # define ocr engine
            ocr_engine = detectors.TesseractOCR(
                l='eng+fas',
                dpi=150,
                psm=12,
                oem=1,
            )
            table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
            table_cell_ocr.vertical_lines = lines[0]
            table_cell_ocr.horizontal_lines = lines[1]
            table_cell_ocr(image=img,
                        plot=MLFLOW_ARTIFACTS_IMAGES_PATH)
        
        elif DETECTION_MODE == DetectionMode.ML_SINGLE_COLUMN_TABLE:
            # smooth image
            gaussian_blur = preprocessors.GaussianImageSmoother(
                border_type=preprocessors.CV2BorderTypes.DEFAULT
            )
            pre_img = gaussian_blur(image=img, kernel_size=3)

            # binarize image
            adaptive_thresh = preprocessors.GaussianAdaptiveThresholder(
                max_value=255,
                adaptive_method=preprocessors.CV2AdaptiveThresholdTypes.GAUSSIAN_C,
                threshold_type=preprocessors.CV2ThresholdTypes.BINARY,
            )
            pre_img = adaptive_thresh(image=pre_img, block_size=11, constant=5,
                                  plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

            # make text blocks as solid blocks
            dilater = preprocessors.Dilate(
                morph_size=3,
            )
            dilated_img = dilater(image=pre_img, iterations=3,
                          plot=MLFLOW_ARTIFACTS_IMAGES_PATH)
            
            # detect lines of table and cells
            contour_line_detector = detectors.ContourLinesDetector(
                cell_threshold=10,
                min_columns=1,
            )
            vertical_lines, horizontal_lines = contour_line_detector(
                image=dilated_img,
                min_solid_height_limit=6,
                max_solid_height_limit=40,
                plot=MLFLOW_ARTIFACTS_IMAGES_PATH
            )

            # define ocr engine
            ocr_engine = detectors.TesseractOCR(
                l='eng',
                dpi=100,
                psm=11,
                oem=1,
            )
            table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
            table_cell_ocr.vertical_lines = vertical_lines
            table_cell_ocr.horizontal_lines = horizontal_lines
            table_cell_ocr(image=pre_img,
                           roi_offset=0,
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
    