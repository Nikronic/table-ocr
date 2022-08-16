# helpers
from ttocr.utils import helpers
import logging


# set logger
logger = logging.getLogger(__name__)


class DetectionMode(helpers.ExtendedEnum):
    """
    Enum for the different detection modes

    Detectors are defined inside the detection module
    i.e. :mod:`ttocr.detection`.
    """

    # deep learning based full table detection (i.e. layout detection)
    DL_FULL_TABLE = 1, True
    # deep learning based single column table detection (i.e. layout detection)
    DL_SINGLE_COLUMN_TABLE = 2, True
    # machine learning based full table detection (e.g. cv2)
    ML_FULL_TABLE = 3
    # machine learning based single column table detection (e.g. cv2)
    ML_SINGLE_COLUMN_TABLE = 4
