__all__ = [
    'ImageColorConverter', 'CV2ImageColorConverterModes', 'CV2ImageColorConverter'
]

# core
from enum import Enum
import numpy as np
import cv2
# helpers
from typing import Any, Optional
import logging


# setup logger
logger = logging.getLogger(__name__)


class ImageColorConverter:
    """Convert image color space

    """

    def __init__(self, mode: Any) -> None:
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)
        
        self.mode = mode

    def __validate_mode(self, mode: Any) -> None:
        """Verifies that ``mode`` exits

        Args:
            mode (Any): mode to verify
        """
        raise NotImplementedError

    def __log(self):
        raise NotImplementedError

    def __call__(self, image: np.ndarray,
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

    def __init__(self, mode: Optional[CV2ImageColorConverterModes] = None) -> None:
        """Initialize ``CV2ImageColorConverter`` with given ``mode``

        Args:
            mode (CV2ImageColorConverterModes): color mode. For more info
                see :class:`CV2ImageColorConverterModes`
        """
        super().__init__(mode)

    def _validate_mode(self, mode: CV2ImageColorConverterModes) -> None:
        """Verifies that ``mode`` exits in ``cv2.COLOR_*``

        Args:
            mode (CV2ImageColorConverterModes): mode to verify
        """
        if mode is None:
            raise ValueError('mode is None. Please provide mode via init or call') 

        if mode not in CV2ImageColorConverterModes:
            raise ValueError(f'Invalid mode: {mode}')

    def _log(self, mode: CV2ImageColorConverterModes) -> None:
        self.logger.info(f'Image is converted to {mode}')

    def __call__(self, image: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        """Convert image color space via ``OpenCV``
        
        Args:
            image (:class:`numpy.ndarray`): image to convert
            mode (CV2ImageColorConverterModes): color mode. For more info
                see :class:`CV2ImageColorConverterModes`
            *args: additional arguments for ``cv2.cvtColor``
            **kwargs: additional keyword arguments for ``cv2.cvtColor``
        """
        # if kwargs provided, override class attributes
        self.mode = kwargs.get('mode', self.mode)

        self._validate_mode(self.mode)
        self._log(self.mode)
        return cv2.cvtColor(image, self.mode.value, *args, **kwargs)
