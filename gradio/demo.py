# core
import numpy as np
# ours
from ttocr.data import io
from ttocr.data import preprocessors
from ttocr.detection import detectors
from ttocr.detection import DetectionMode
from ttocr.version import VERSION as TTOCR_VERSION
# devops
import mlflow
# demo
import gradio as gr
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
MLFLOW_EXPERIMENT_NAME = f'Gradio - {TTOCR_VERSION}'
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_TAGS = {
    'stage': 'dev'  # dev, beta, production
}
mlflow.set_tags(MLFLOW_TAGS)

logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

# gradio interface
def predict(
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

# build gradio interface given dataframe dtype
inputs: List[gr.components.Component] = [
    # input image
    gr.Image(
        label='The image that needs to be OCRed',
    ), 
    gr.Checkbox(
        label='Single Column?',
        value=True,
    ),

    # preprocessing for ML_FULL_TABLE
    gr.Slider(
        label='Canny threshold1',
        minimum=0,
        maximum=255,
        step=5,
        value=50,
    ),
    gr.Slider(
        label='Canny threshold2',
        minimum=0,
        maximum=255,
        step=5,
        value=200,
    ),
    gr.Slider(
        label='Canny aperture size',
        minimum=3,
        maximum=15,
        step=2,
        value=3,
    ),
    gr.Checkbox(
        label='Canny L2 gradient',
        value=False,
    ),
    gr.Number(
        label='Hough lines minimum line length',
        precision=0,
        value=40,
    ),
    gr.Number(
        label='Hough lines maximum line gap',
        precision=0,
        value=10,
    ),

    # preprocessing for ML_SINGLE_COLUMN_TABLE
    gr.Slider(
        label='Smoothing filter kernel size',
        minimum=3,
        maximum=15,
        step=2,
        value=3,
    ),
    gr.Slider(
        label='Adaptive thresholding block size',
        minimum=3,
        maximum=21,
        step=2,
        value=11,
    ),
    gr.Slider(
        label='Adaptive thresholding constant',
        minimum=1,
        maximum=10,
        step=1,
        value=5,
    ),
    gr.Slider(
        label='Structuring element size of dilation',
        minimum=1,
        maximum=15,
        step=2,
        value=3,
    ),
    gr.Number(
        label='Number of dilation iterations',
        precision=0,
        value=2,
    ),
    gr.Number(
        label='Line cell threshold',
        precision=0,
        value=10,
    ),
    gr.Number(
        label='Minimum cell height',
        precision=0,
        value=6,
    ),
    gr.Number(
        label='Maximum cell height',
        precision=0,
        value=40,
    ),
    gr.Slider(
        label='ROI offset (margin)',
        minimum=0,
        maximum=10,
        step=1,
        value=0,
    ),

    # common
    gr.Textbox(
        label='Languages',
        value='eng',
        lines=1,
        max_lines=1,
        placeholder='eng+fas+[LANG_3CHAR_CODE]',
    ),
    gr.Number(
        label='OCR DPI',
        precision=0,
        value=150,
    ),
    gr.Dropdown(
        label='OCR PSM',
        choices=[
            '3', '6', '11', '12'
        ],
        value='6',
    ),
    gr.Dropdown(
        label='OCR PSM',
        choices=[
            '1', '2', '3', '4'
        ],
        value='1',
    ),
]


# prediction probability output
outputs: gr.components.Component = [
    gr.Dataframe(
        label='OCR output',
        type='array',
        max_rows=None,
        max_cols=None,
        wrap=True,
    ),
]

title = 'TourTableOCR'
description = 'OCRing weirdge tables'

# close all Gradio instances
gr.close_all()

# launch gradio
gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    flagging_dir='gradio/flags',
    analytics_enabled=True,
).launch(debug=True, enable_queue=True, server_port=7860)
