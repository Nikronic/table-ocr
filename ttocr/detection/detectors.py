__all__ = [
    # abstract classes
    'Detector', 'EdgeDetector', 'LineDetector'
    # concrete classes
    'CannyEdgeDetector', 'ProbabilisticHoughLinesDetector', 'NaiveHoughLinesDetector'
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

class LineDirection(Enum):
    """Enum for line direction

    Notes:
        This is used for line detection such as :class:`LineDetector`
        and any class that subclasses it
    """
    HORIZONTAL = 0
    VERTICAL = 1


class LineDetector(Detector):
    """Detect lines in an image

    User should subclass this and implement the :meth:`detect` method
    based on their detector algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.horizontal_lines: List[np.ndarray] = []
        self.vertical_lines: List[np.ndarray] = []

    def __log(self):
        self.logger.info(
            f'{self.__class__.__name__} line detection is performed')

    def _reset_lines(self):
        self.horizontal_lines: List[np.ndarray] = []
        self.vertical_lines: List[np.ndarray] = []

    @staticmethod
    def _horizontal(line: np.ndarray) -> LineDirection:
        """Check if line is horizontal

        Args:
            line (:class:`numpy.ndarray`): line to check

        Returns:
            LineDirection: Direction of line
        """
        return line[0] == line[2]
    
    @staticmethod
    def _vertical(line: np.ndarray) -> LineDirection:
        """Check if line is vertical

        Args:
            line (:class:`numpy.ndarray`): line to check

        Returns:
            LineDirection: Direction of line
        """
        return line[1] == line[3]
    
    def _find_line_direction(self, line: np.ndarray) -> LineDirection:
        """Find direction of line

        Args:
            line (:class:`numpy.ndarray`): line to check

        Returns:
            LineDirection: Direction of line
        """
        if self._horizontal(line):
            return LineDirection.HORIZONTAL
        elif self._vertical(line):
            return LineDirection.VERTICAL
        else:
            self.logger.warning(f'Line {line} is not horizontal or vertical')
    
    def get_vertical_horizontal_lines(self,
            lines: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Assigns lines to either direction of vertical or horizontal

        This method also fills attributes :attr:`horizontal_lines` and
        :attr:`vertical_lines` with the lines assigned.

        Args:
            lines (List[:class:`numpy.ndarray`]): list of lines detected by detector
        
        Returns:
            Tuple[List[:class:`numpy.ndarray`], List[:class:`numpy.ndarray`]]:
                tuple of list of vertical and horizontal lines in order
        """
        vertical_lines: List[np.ndarray] = []
        horizontal_lines: List[np.ndarray] = []
        for line in lines:
            direction = self._find_line_direction(line)
            if direction == LineDirection.VERTICAL:
                vertical_lines.append(line)
            elif direction == LineDirection.HORIZONTAL:
                horizontal_lines.append(line)
        return vertical_lines, horizontal_lines


    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class ProbabilisticHoughLinesDetector(LineDetector):
    """Detect lines in an image using probabilistic Hough transform (cv2.HoughLinesP_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.1.0/d9/d61/tutorial_hough_lines.html>`_.

    .. _cv2.HoughLinesP: https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    """

    def __init__(self) -> None:
        super().__init__()

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Detect lines using ``cv2.HoughLinesP``

        Args:
            image (:class:`numpy.ndarray`): image to detect lines
            args: arguments for ``cv2.HoughLinesP``
            kwargs: keyword arguments for ``cv2.HoughLinesP``

        Returns:
            :class:`numpy.ndarray`: image with detected lines
        """
        return cv2.HoughLinesP(image, *args, **kwargs)


class NaiveHoughLinesDetector(LineDetector):
    """Detect lines in an image using naive Hough transform (cv2.HoughLines_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.1.0/d9/d61/tutorial_hough_lines.html>`_.

    .. _cv2.HoughLines: https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
    """

    def __init__(self) -> None:
        super().__init__()

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Detect lines using ``cv2.HoughLines``

        Args:
            image (:class:`numpy.ndarray`): image to detect lines
            args: arguments for ``cv2.HoughLines``
            kwargs: keyword arguments for ``cv2.HoughLines``

        Returns:
            :class:`numpy.ndarray`: image with detected lines
        """
        return cv2.HoughLinesP(image, *args, **kwargs)
