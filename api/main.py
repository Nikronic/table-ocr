# core
import numpy as np
import cv2
# ours
from ttocr.data import io
from ttocr.data import preprocessors
from ttocr.detection import detectors
from ttocr.detection import DetectionMode
from ttocr.api import models as api_models
from ttocr.version import VERSION as TTOCR_VERSION
# api
import fastapi
import uvicorn
import base64
# devops
import mlflow
# helpers
from typing import List
from pathlib import Path
import shutil
import logging
import sys


# configure logging
VERBOSE = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(VERBOSE)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)

# setup dirs for mlflow artifacts (logs, configs, etc)
mlflow.set_tracking_uri('http://localhost:5000')
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
logger_handler = logging.FileHandler(
    filename=MLFLOW_ARTIFACTS_LOGS_PATH / 'main.log',
    mode='w'
)
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
MLFLOW_EXPERIMENT_NAME = f'FastAPI service - {TTOCR_VERSION}'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_TAGS = {
    'stage': 'beta'  # dev, beta, production
}
mlflow.set_tags(MLFLOW_TAGS)

logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

title = 'TourTableOCR'
description = 'OCRing weirdge tables'

# predict on submit
def _predict(
    image: np.ndarray,
    mode: bool = True,

    # preprocessing for ML_TABLE
    canny_threshold1: float = 50,
    canny_threshold2: float = 200,
    canny_aperture_size: int = 3,
    canny_L2_gradient: bool = False,

    hough_min_line_length: int = 50,
    hough_max_line_gap: int = 10,

    # preprocessing for ML_SINGLE_COLUMN_TABLE
    smooth_kernel_size: int = 3,

    thresh_block_size: int = 11,
    thresh_c: int = 5,

    dilate_morph_size: int = 3,
    dilation_iterations: int = 3,

    contour_line_cell_threshold: int = 10,
    contour_min_solid_height_limit: int = 6,
    contour_max_solid_height_limit: int = 40,

    roi_offset: int = 0,

    # common
    ocr_lang: str = 'eng',
    ocr_dpi: int = 150,
    ocr_psm: int = 6,
    ocr_oem: int = 1,


) -> List[str]:
    # convert gradio interface type to ttocr type
    ocr_psm = int(ocr_psm)
    ocr_oem = int(ocr_oem)

    # choose table detection method
    if mode:
        DETECTION_MODE = DetectionMode.ML_SINGLE_COLUMN_TABLE
    else:
        DETECTION_MODE = DetectionMode.ML_FULL_TABLE
    
    # get image
    img = image

    # convert color to gray
    color_converter = preprocessors.CV2ImageColorConverter(
        mode=preprocessors.CV2ImageColorConverterModes.BGR2GRAY
    )
    img = color_converter(image=img)

    if DETECTION_MODE == DetectionMode.ML_FULL_TABLE:
        # detect canny edges
        canny_detector = detectors.CannyEdgeDetector(
            threshold1=canny_threshold1,
            threshold2=canny_threshold2,
            aperture_size=canny_aperture_size,
            L2_gradient=canny_L2_gradient
        )
        canny_edges = canny_detector(image=img,
                                    plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

        # detect lines
        line_detector = detectors.ProbabilisticHoughLinesDetector(
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            min_line_length=hough_min_line_length,
            max_line_gap=hough_max_line_gap
        )
        lines = line_detector(image=canny_edges)
        
        # define ocr engine
        ocr_engine = detectors.TesseractOCR(
            l=ocr_lang,
            dpi=ocr_dpi,  # 300
            psm=ocr_psm,  # 6
            oem=ocr_oem,  # 1
        )
        table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
        table_cell_ocr.vertical_lines = lines[0]
        table_cell_ocr.horizontal_lines = lines[1]
        texts = table_cell_ocr(
            image=img,
            plot=MLFLOW_ARTIFACTS_IMAGES_PATH
        )
        return texts
    
    elif DETECTION_MODE == DetectionMode.ML_SINGLE_COLUMN_TABLE:
        # smooth image
        gaussian_blur = preprocessors.GaussianImageSmoother(
            border_type=preprocessors.CV2BorderTypes.DEFAULT
        )
        pre_img = gaussian_blur(image=img, kernel_size=smooth_kernel_size)

        # binarize image
        adaptive_thresh = preprocessors.GaussianAdaptiveThresholder(
            max_value=255,
            adaptive_method=preprocessors.CV2AdaptiveThresholdTypes.GAUSSIAN_C,
            threshold_type=preprocessors.CV2ThresholdTypes.BINARY,
        )
        pre_img = adaptive_thresh(image=pre_img,
                                block_size=thresh_block_size,
                                constant=thresh_c,
                                plot=MLFLOW_ARTIFACTS_IMAGES_PATH)

        # make text blocks as solid blocks
        dilater = preprocessors.Dilate(
            morph_size=dilate_morph_size,
        )
        dilated_img = dilater(image=pre_img, iterations=dilation_iterations,
                        plot=MLFLOW_ARTIFACTS_IMAGES_PATH)
        
        # detect lines of table and cells
        contour_line_detector = detectors.ContourLinesDetector(
            cell_threshold=contour_line_cell_threshold,
            min_columns=1,
        )
        vertical_lines, horizontal_lines = contour_line_detector(
            image=dilated_img,
            min_solid_height_limit=contour_min_solid_height_limit,
            max_solid_height_limit=contour_max_solid_height_limit,
            plot=MLFLOW_ARTIFACTS_IMAGES_PATH
        )

        # define ocr engine
        ocr_engine = detectors.TesseractOCR(
            l=ocr_lang,
            dpi=ocr_dpi,  # 150
            psm=ocr_psm,  # 11
            oem=ocr_oem,  # 1
        )
        table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
        table_cell_ocr.vertical_lines = vertical_lines
        table_cell_ocr.horizontal_lines = horizontal_lines
        texts = table_cell_ocr(image=pre_img,
                            roi_offset=roi_offset,
                            lot=MLFLOW_ARTIFACTS_IMAGES_PATH)
        return texts


# instantiate fast api app
app = fastapi.FastAPI()


@app.post("/predict/", response_model=api_models.PredictionResponse)
async def predict(
    conf: api_models.Payload,
    ):
    # if file.content_type.startswith('image/') is False:
    #     raise fastapi.HTTPException(
    #         status_code=400,
    #         detail=f'File \'{file.filename}\' is not an image.')    

    try:
        # contents = await file.read()
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(conf.file), np.uint8),
            cv2.IMREAD_COLOR
        )
        ocr_result = _predict(
            image=image,
            mode=conf.mode,

            # preprocessing for ML_TABLE
            canny_threshold1=conf.canny_threshold1,
            canny_threshold2=conf.canny_threshold2,
            canny_aperture_size=conf.canny_aperture_size,
            canny_L2_gradient=conf.canny_L2_gradient,

            hough_min_line_length=conf.hough_min_line_length,
            hough_max_line_gap=conf.hough_max_line_gap,

            # preprocessing for ML_SINGLE_COLUMN_TABLE
            smooth_kernel_size=conf.smooth_kernel_size,

            thresh_block_size=conf.thresh_block_size,
            thresh_c=conf.thresh_c,

            dilate_morph_size=conf.dilate_morph_size,
            dilation_iterations=conf.dilation_iterations,

            contour_line_cell_threshold=conf.contour_line_cell_threshold,
            contour_min_solid_height_limit=conf.contour_min_solid_height_limit,
            contour_max_solid_height_limit=conf.contour_max_solid_height_limit,

            roi_offset=conf.roi_offset,

            # common
            ocr_lang=conf.ocr_lang,
            ocr_dpi=conf.ocr_dpi,
            ocr_psm=conf.ocr_psm,
            ocr_oem=conf.ocr_oem,
        )
        
        logger.info('OCR finished')
        return {
            'ocr_result': ocr_result,
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))

