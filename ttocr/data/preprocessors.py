__all__ = [
    'ImageColorConverter', 'CV2ImageColorConverterModes'
]

# core
from enum import Enum
import numpy as np
import cv2
# helpers
from typing import Any
import logging


# setup logger
logger = logging.getLogger(__name__)


class ImageColorConverter:
    """Convert image color space

    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)
        pass

    def __validate_mode(self, mode: Any) -> None:
        """Verifies that ``mode`` exits

        Args:
            mode (Any): mode to verify
        """
        raise NotImplementedError

    def __log(self):
        raise NotImplementedError

    def __call__(self, image: np.ndarray, mode: Any,
                 *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class CV2ImageColorConverterModes(Enum):
    """Enum for color conversion modes in ``cv2.COLOR_*``
    
    """
    BGR2RGB = cv2.COLOR_BGR2RGB
    RGB2BGR = cv2.COLOR_RGB2BGR
    BGR2GRAY = cv2.COLOR_BGR2GRAY
    GRAY2BGR = cv2.COLOR_GRAY2BGR
    BGR2HSV = cv2.COLOR_BGR2HSV
    HSV2BGR = cv2.COLOR_HSV2BGR
    BGR2HLS = cv2.COLOR_BGR2HLS
    HLS2BGR = cv2.COLOR_HLS2BGR
    BGR2Lab = cv2.COLOR_BGR2Lab
    Lab2BGR = cv2.COLOR_Lab2BGR
    BGR2Luv = cv2.COLOR_BGR2Luv
    Luv2BGR = cv2.COLOR_Luv2BGR
    BGR2YUV = cv2.COLOR_BGR2YUV
    YUV2BGR = cv2.COLOR_YUV2BGR
    BGR2YCrCb = cv2.COLOR_BGR2YCrCb
    YCrCb2BGR = cv2.COLOR_YCrCb2BGR
    BGR2XYZ = cv2.COLOR_BGR2XYZ
    XYZ2BGR = cv2.COLOR_XYZ2BGR


class CV2ImageColorConverter(ImageColorConverter):
    """Convert image color space via ``OpenCV``
    
    """
    def __init__(self) -> None:
        super().__init__()

    def __validate_mode(self, mode: CV2ImageColorConverterModes) -> None:
        """Verifies that ``mode`` exits in ``cv2.COLOR_*``

        Args:
            mode (CV2ImageColorConverterModes): mode to verify
        """
        all_modes = CV2ImageColorConverterModes.__members__.items()
        if mode not in all_modes:
            raise ValueError(f'Invalid mode: {mode}')

    def __log(self, mode: CV2ImageColorConverterModes) -> None:
        self.logger.info(f'Image is converted to {mode}')

    def __call__(self, image: np.ndarray,
                 mode: CV2ImageColorConverterModes,
                 *args, **kwargs) -> np.ndarray:
        self.__validate_mode(mode)
        self.__log(mode)
        return cv2.cvtColor(image, mode, *args, **kwargs)
