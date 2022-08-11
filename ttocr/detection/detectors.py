__all__ = [
    # abstract classes
    'Detector', 'EdgeDetector',
    # concrete classes
    'CannyEdgeDetector', 
]

# core
from enum import Enum
from typing import List, Tuple
import numpy as np
import cv2
# helpers
import logging


# setup logger
logger = logging.getLogger(__name__)


class Detector:
    """Base class for all detectors such as edge and line detectors

    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)

    def __log(self):
        raise NotImplementedError

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self.__log()
        return self.detect(image, *args, **kwargs)


class EdgeDetector(Detector):
    """Detect edges in an image

    """

    def __init__(self) -> None:
        super().__init__()

    def __log(self):
        self.logger.info(
            f'{self.__class__.__name__} edge detection is performed')

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class CannyEdgeDetector(EdgeDetector):
    """Detect edges in an image using Canny algorithm (cv2.Canny_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>`_. 
    
    .. _cv2.Canny: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    """

    def __init__(self) -> None:
        super().__init__()

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Detect edges using ``cv2.Canny``

        Args:
            image (:class:`numpy.ndarray`): image to detect edges
            args: arguments for cv2.Canny_
            kwargs: keyword arguments for cv2.Canny_

        Returns:
            :class:`numpy.ndarray`: image with detected edges 
        """
        return cv2.Canny(image, *args, **kwargs)

