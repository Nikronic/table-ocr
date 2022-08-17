__all__ = [
    'ImageColorConverter', 'CV2ImageColorConverterModes', 'CV2ImageColorConverter'
]

# core
from enum import Enum
import numpy as np
import cv2
# helpers
from typing import Any, Optional, Tuple
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


class ImageSmoother:
    """Abstract class for implementing different smoothing methods
    """

    def __init__(self, kernel_size: Optional[int] = None) -> None:
        """Initialize ``ImageSmoother`` with given ``kernel_size``
        
        Args:
            kernel_size (int): kernel size for smoothing
        """
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)
        self.kernel_size = kernel_size
    
    def _get_class_attributes(self) -> dict:
        return dict(self.__dict__)
    
    def _log(self, *args, **kwargs):
        self.logger.info(
            f'{self.__class__.__name__} image smoothing is performed'
            f' with kwargs: {kwargs}'
        )

    def smooth(self, image: np.ndarray,
               *args, **kwargs) -> np.ndarray:
        """Smooth image via given method
        
        Args:
            image (:class:`numpy.ndarray`): image to smooth
            *args: additional arguments for smoother
            **kwargs: additional keyword arguments for smoother
        """
        raise NotImplementedError

    def __call__(self, image: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class CV2BorderTypes(Enum):
    """Enum for border types in ``cv2.BORDER_*``

    Notes:
        For more information, see https://docs.opencv.org/4.6.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    """
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    REFLECT_101 = cv2.BORDER_REFLECT_101
    WRAP = cv2.BORDER_WRAP
    TRANSPARENT = cv2.BORDER_TRANSPARENT
    ISOLATED = cv2.BORDER_ISOLATED
    DEFAULT = cv2.BORDER_DEFAULT


class GaussianImageSmoother(ImageSmoother):
    """Blurs an image via Open CVs cv2.GaussianBlur_

    Notes:
        For more info about the algorithm see https://docs.opencv.org/4.6.0/d4/d13/tutorial_py_filtering.html

    .. _cv2.GaussianBlur: * https://docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    """

    def __init__(self,
                 kernel_size: Optional[int] = None,
                 border_type: Optional[CV2BorderTypes] = 0) -> None:
        """Initialize ``GaussianImageSmoother`` with given ``kernel_size``
        
        Args:
            kernel_size (int): kernel size for smoothing
        """
        super().__init__(kernel_size)
        self.border_type = border_type

    def smooth(self, image: np.ndarray,
               *args, **kwargs) -> np.ndarray:
        """Smooth image via given method
        
        Args:
            image (:class:`numpy.ndarray`): image to smooth
            *args: additional arguments for smoother
            **kwargs: additional keyword arguments for smoother
        """
        return cv2.GaussianBlur(image, *args, **kwargs)

    def __call__(self, image: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        # if kwargs provided, override class attributes
        self.border_type = kwargs.get('border_type', self.border_type)
        self.kernel_size = kwargs.get('kernel_size', self.kernel_size)

        # logging
        self._log(self._get_class_attributes())

        smoothed = self.smooth(
            image=image,
            kernel_size=self.kernel_size,
            border_type=self.border_type
        )
        
        return smoothed
