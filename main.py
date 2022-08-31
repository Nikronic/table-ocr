# core
import numpy as np
# ours: data
from ttocr.data import io
from ttocr.data import preprocessors
# ours: detection
from ttocr.detection import DetectionMode
from ttocr.detection import detectors
from ttocr.utils import loggers
# ours: utils
from ttocr.version import VERSION as TTOCR_VERSION
# devops
import mlflow
# benchmark
from pyinstrument import Profiler
# helpers
from typing import List, Optional
from pathlib import Path
import logging
import shutil
import sys



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

    return texts


if __name__ == '__main__':
    # globals
    SEED = 8569
    VERBOSE = logging.DEBUG
    FLAG: bool = False
    DEVICE = 'cuda'

    # configure MLFlow tracking remote server
    #  see `mlflow-server.sh` for port and hostname. Since
    #  we are running locally, we can use the default values.
    mlflow.set_tracking_uri('http://0.0.0.0:5000')

    # configure logging
    VERBOSE = logging.DEBUG
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
    profiler = Profiler()

    # log experiment configs
    MLFLOW_EXPERIMENT_NAME = f'Fix#10 - {TTOCR_VERSION}'
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.start_run()

    logger.info(f'MLflow experiment name: {MLFLOW_EXPERIMENT_NAME}')
    logger.info(f'MLflow experiment id: {mlflow.active_run().info.run_id}')

    try:
        # create mlflow artifact instance for each run
        logger.create_artifact_instance()

        # choose table detection method
        DETECTION_MODE = DetectionMode.ML_SINGLE_COLUMN_TABLE
        logger.info(f'Detection mode: {DETECTION_MODE}')

        # read image
        filename = 'sample/orig/03-col-with-border.png'
        img_reader = io.CV2ImageReader()
        img = img_reader(filename)

        # benchmark
        profiler.start()
        result = _predict(image=img, flag=FLAG)
        profiler.stop()
        # save benchmark to disk
        profiler_output = profiler.output_html()
        with open(
            logger.MLFLOW_ARTIFACTS_LOGS_PATH / f'{mlflow.active_run().info.run_id}.html',
            'w') as fp:
            fp.write(profiler_output)
            fp.close()

    except Exception as e:
        logger.exception(e)
        raise e
    
    # cleanup code
    finally:
        logger.info(f'artifacts saved in MLflow artifacts directory.')
        mlflow.log_artifacts(MLFLOW_ARTIFACTS_BASE_PATH)
        # delete redundant logs, files that are logged as artifact
        shutil.rmtree(MLFLOW_ARTIFACTS_BASE_PATH)
        # end mlflow run
        mlflow.end_run()
    