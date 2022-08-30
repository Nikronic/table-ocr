# core
import numpy as np
import cv2
# ours
from ttocr.data import io
from ttocr.data import preprocessors
from ttocr.detection import detectors
from ttocr.detection import DetectionMode
from ttocr.utils import loggers
from ttocr.api import models as api_models
from ttocr.api import apps as api_apps
from ttocr.version import VERSION as TTOCR_VERSION
# api
import fastapi
import uvicorn
# devops
import mlflow
# helpers
from typing import List, Optional
from pathlib import Path
import argparse
import shutil
import logging
import sys


# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str,
                    help='mlflow experiment name for logging',
                    default='',
                    required=True)
parser.add_argument('--verbose', type=str,
                    help='logging verbosity level.',
                    choices=['debug', 'info'],
                    default='info',
                    required=True)
parser.add_argument('--bind', type=str,
                    help='ip address of host',
                    default='0.0.0.0',
                    required=True)
parser.add_argument('--mlflow_port', type=int,
                    help='port of mlflow tracking',
                    default=5000,
                    required=True)
parser.add_argument('--gunicorn_port', type=int,
                    help='port used for creating gunicorn',
                    default=8000,
                    required=True)
parser.add_argument('--workers', type=int,
                    help='number of works used by gunicorn',
                    default=1,
                    required=True)
args = parser.parse_args()

# configure logging
VERBOSE = logging.DEBUG if args.verbose == 'debug' else logging.INFO
MLFLOW_ARTIFACTS_BASE_PATH: Path = Path('artifacts')
if MLFLOW_ARTIFACTS_BASE_PATH.exists():
    shutil.rmtree(MLFLOW_ARTIFACTS_BASE_PATH)
__libs = ['ttocr']
logger = loggers.Logger(
    name=__name__,
    level=VERBOSE,
    mlflow_artifacts_base_path=MLFLOW_ARTIFACTS_BASE_PATH,
    libs=__libs
)

# run mlflow tracking server
mlflow.set_tracking_uri(f'http://{args.bind}:{args.mlflow_port}')

# log experiment configs
if args.experiment_name == '':
    MLFLOW_EXPERIMENT_NAME = f'{TTOCR_VERSION}'
else:
    MLFLOW_EXPERIMENT_NAME = f'{args.experiment_name} - {TTOCR_VERSION}'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.start_run()

logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

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

    flag: bool = False
) -> List[str]:
    # output
    texts: Optional[List[List[str]]] = None

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
        canny_edges = canny_detector(
            image=img,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None
        )

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
        result = table_cell_ocr(
            image=img,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None
        )
        # drop annotation of OCRed image in debug mode (2nd element is image)
        texts = result[0] if flag else result

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
        pre_img = adaptive_thresh(
            image=pre_img,
            block_size=thresh_block_size,
            constant=thresh_c,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None
        )

        # make text blocks as solid blocks
        dilater = preprocessors.Dilate(
            morph_size=dilate_morph_size,
        )
        dilated_img = dilater(
            image=pre_img,
            iterations=dilation_iterations,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None)

        # detect lines of table and cells
        contour_line_detector = detectors.ContourLinesDetector(
            cell_threshold=contour_line_cell_threshold,
            min_columns=1,
        )
        vertical_lines, horizontal_lines = contour_line_detector(
            image=dilated_img,
            min_solid_height_limit=contour_min_solid_height_limit,
            max_solid_height_limit=contour_max_solid_height_limit,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None
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
        result = table_cell_ocr(
            image=pre_img,
            roi_offset=roi_offset,
            plot=logger.MLFLOW_ARTIFACTS_IMAGES_PATH if flag else None
        )
        # drop annotation of OCRed image in debug mode (2nd element is image)
        texts = result[0] if flag else result
    # if need to be flagged, save as artifact
    if flag:
        logger.info(f'artifacts saved in MLflow artifacts directory.')
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)
    return texts


# instantiate fast api app
app = fastapi.FastAPI()


@app.post("/predict/", response_model=api_models.PredictionResponse)
async def predict(
    conf: api_models.Payload,
    file: fastapi.UploadFile = fastapi.File(...)
):
    if file.content_type.startswith('image/') is False:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
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
            flag=False
        )

        logger.info('OCR finished')
        return {
            'ocr_result': ocr_result,
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@app.post("/flag/", response_model=api_models.PredictionResponse)
async def flag(
    conf: api_models.Payload,
    file: fastapi.UploadFile = fastapi.File(...)
):
    if file.content_type.startswith('image/') is False:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f'File \'{file.filename}\' is not an image.')

    # create new instance of mlflow artifact
    logger.create_artifact_instance()

    try:
        contents = await file.read()
        image = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
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
            flag=True
        )

        logger.info('OCR finished')
        return {
            'ocr_result': ocr_result,
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise fastapi.HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    options = {
        'bind': f'{args.bind}:{args.gunicorn_port}',
        'workers': args.workers,
        'worker_class': 'uvicorn.workers.UvicornWorker'
    }
    api_apps.StandaloneApplication(app=app, options=options).run()
