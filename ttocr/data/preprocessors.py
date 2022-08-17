__all__ = [
    'ImageColorConverter', 'CV2ImageColorConverterModes', 'CV2ImageColorConverter',

    'ImageSmoother', 'GaussianImageSmoother', 'CV2BorderTypes',

    'ImageThresholder', 'CV2ThresholdTypes', 'GaussianAdaptiveThresholder',
    'OtsuThresholder',
]

# core
import numpy as np
import cv2
# helpers
from typing import Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
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
                 border_type: CV2BorderTypes = CV2BorderTypes.DEFAULT) -> None:
        """Initialize ``GaussianImageSmoother`` with given ``kernel_size``
        
        Args:
            kernel_size (int): kernel size for smoothing
            border_type (CV2BorderTypes): border type for smoothing (kernel operation)
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
        self._log(**self._get_class_attributes())

        smoothed = self.smooth(
            image=image,
            ksize=(self.kernel_size, self.kernel_size),
            sigmaX=0,  # lets compute sigma based on kernel size (ie = 0)
            borderType = self.border_type.value
        )
        
        return smoothed


class ImageThresholder:
    """Abstract class for implementing different thresholding methods
    """

    def __init__(self) -> None:
        """Initialize ``ImageThresholder``
        """
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)

    def _get_class_attributes(self) -> dict:
        return dict(self.__dict__)
    
    def _log(self, *args, **kwargs):
        self.logger.info(
            f'{self.__class__.__name__} image thresholding is performed'
            f' with kwargs: {kwargs}'
        )

    def threshold(self, image: np.ndarray,
                  *args, **kwargs) -> np.ndarray:
        """Threshold image via given method
        
        Args:
            image (:class:`numpy.ndarray`): image to threshold
            *args: additional arguments for thresholder
            **kwargs: additional keyword arguments for thresholder
        """
        raise NotImplementedError

    def __call__(self, image: np.ndarray,
                 *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class CV2ThresholdTypes(Enum):
    """Enum for threshold types in ``cv2.THRESH_*``
    
    Notes:
        For more information, see https://docs.opencv.org/4.6.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    """
    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
    TOZERO_INV = cv2.THRESH_TOZERO_INV
    OTSU = cv2.THRESH_OTSU


class CV2AdaptiveThresholdTypes(Enum):
    """Enum for adaptive threshold types in ``cv2.ADAPTIVE_THRESH_*``
    
    Notes:
        For more information, see https://docs.opencv.org/4.6.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    """
    MEAN_C = cv2.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN_C = cv2.ADAPTIVE_THRESH_GAUSSIAN_C


class GaussianAdaptiveThresholder(ImageThresholder):
    """Binarizes an image in an Gaussian adaptive manner via cv2.adaptiveThreshold_

    Notes:
        For more info about the algorithm, see https://docs.opencv.org/4.6.0/d7/d4d/tutorial_py_thresholding.html
    
    .. _cv2.adaptiveThreshold: https://docs.opencv.org/4.6.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    """

    def __init__(self,
                 max_value: Optional[int] = None,
                 adaptive_method: Optional[CV2AdaptiveThresholdTypes] = None,
                 threshold_type: Optional[CV2ThresholdTypes] = None,
                 block_size: Optional[int] = None,
                 constant: Optional[float] = None) -> None:
        """Initialize ``GaussianAdaptiveThresholder``
        
        Args:
            max_value (int): maximum value for pixels in the image
            adaptive_method (CV2AdaptiveThresholdTypes): adaptive thresholding algorithm
            threshold_type (CV2ThresholdTypes): threshold type for thresholding.
                Can be only one of ``CV2ThresholdTypes.BINARY`` and ``CV2ThresholdTypes.BINARY_INV``,
            block_size (int): window size for calculating threshold
            constant (float): constant subtracted from mean and weighted mean
        """
        super().__init__()
        self.max_value = max_value
        self.adaptive_method = adaptive_method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.constant = constant
    
    @staticmethod
    def _check_threshold_type(type) -> None:
        """Checks if threshold type is valid
        """
        if type not in [CV2ThresholdTypes.BINARY, CV2ThresholdTypes.BINARY_INV]:
            raise ValueError(
                f'threshold type must be one of'
                f' {CV2ThresholdTypes.BINARY} and'
                f' {CV2ThresholdTypes.BINARY_INV}'
            )

    def threshold(self, image: np.ndarray,
                  *args, **kwargs) -> np.ndarray:
        """Threshold image via given method
        
        Args:
            image (:class:`numpy.ndarray`): image to threshold
            *args: additional arguments for thresholder
            **kwargs: additional keyword arguments for thresholder
        """
        return cv2.adaptiveThreshold(image, *args, **kwargs)

    def __call__(self, image: np.ndarray,
                 plot: Optional[Path] = None,
                 *args, **kwargs) -> np.ndarray:
        # if kwargs provided, override class attributes
        self.max_value = kwargs.get('max_value', self.max_value)
        self.adaptive_method = kwargs.get('adaptive_method', self.adaptive_method)
        self.threshold_type = kwargs.get('threshold_type', self.threshold_type)
        self.block_size = kwargs.get('block_size', self.block_size)
        self.constant = kwargs.get('constant', self.constant)

        # check type
        self._check_threshold_type(self.threshold_type)

        thresholded = self.threshold(
            image=image,
            maxValue=self.max_value,
            adaptiveMethod=self.adaptive_method.value,
            thresholdType=self.threshold_type.value,
            blockSize=self.block_size,
            C=self.constant
        )

        # logging
        self._log(**self._get_class_attributes())
        if plot is not None:
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(thresholded, cmap='gray')
            plt.savefig(plot / 'gaussian_adaptive_thresh.png')
            plt.close(fig)
        
        return thresholded


class OtsuThresholder(ImageThresholder):
    """Binarizes an image in an Otsu adaptive manner via cv2.threshold_
    
    Notes:
        For more info about the algorithm, see https://docs.opencv.org/4.6.0/d7/d4d/tutorial_py_thresholding.html
    
    .. _cv2.threshold: https://docs.opencv.org/4.6.0/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
    """

    def __init__(self,
                 threshold_value: Optional[float] = None,
                 max_value: Optional[int] = None,
                 threshold_type: Optional[CV2ThresholdTypes] = None
                 ) -> None:
        """Initialize ``OtsuThresholder``
        
        Args:
            threshold_value (float): threshold value
            max_value (int): maximum value to use with threshold types
                ``CV2ThresholdTypes.BINARY`` and ``CV2ThresholdTypes.BINARY_INV``
                from :class:`CV2ThresholdTypes`
            threshold_type (CV2ThresholdTypes): threshold type for thresholding
        """
        super().__init__()
        self.threshold_value = threshold_value
        self.max_value = max_value
        self.threshold_type = threshold_type
    
    @staticmethod
    def _check_threshold_type(type) -> None:
        """Checks if threshold type is valid
        """
        if type not in [CV2ThresholdTypes.BINARY, CV2ThresholdTypes.BINARY_INV]:
            raise ValueError(
                f'threshold type must be one of'
                f' {CV2ThresholdTypes.BINARY} and'
                f' {CV2ThresholdTypes.BINARY_INV}'
            )

    def threshold(self, image: np.ndarray,
                  *args, **kwargs) -> np.ndarray:
        """Threshold image via given method
        
        Args:
            image (:class:`numpy.ndarray`): image to threshold
            *args: additional arguments for thresholder
            **kwargs: additional keyword arguments for thresholder
        """
        return cv2.threshold(image, *args, **kwargs)
    
    def __call__(self, image: np.ndarray,
                 plot: Optional[Path] = None,
                 *args, **kwargs) -> np.ndarray:
        # if kwargs provided, override class attributes
        self.threshold_value = kwargs.get('threshold_value', self.threshold_value)
        self.max_value = kwargs.get('max_value', self.max_value)
        self.threshold_type = kwargs.get('threshold_type', self.threshold_type)

        # check type
        self._check_threshold_type(self.threshold_type)

        thresholded = self.threshold(
            image=image,
            thresh=self.threshold_value,
            maxValue=self.max_value,
            type=self.threshold_type.value + CV2ThresholdTypes.OTSU.value
        )

        # logging
        self._log(**self._get_class_attributes())
        if plot is not None:
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(thresholded, cmap='gray')
            plt.savefig(plot / 'otsu_adaptive_thresh.png')
            plt.close(fig)
        
        return thresholded
