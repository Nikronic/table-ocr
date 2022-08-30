__all__ = [
    # abstract classes
    'Detector', 'EdgeDetector', 'LineDetector', 'OCR',
    # concrete classes
    'CannyEdgeDetector', 'ProbabilisticHoughLinesDetector', 'NaiveHoughLinesDetector',
    'TesseractOCR', 'TableCellDetector', 'ContourLinesDetector'
]

# core
import pytesseract
import numpy as np
import cv2
# ours: helpers
from ttocr.utils import visualizers
# helpers
from typing import List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import logging


# setup logger
logger = logging.getLogger(__name__)


class DetectorBase:
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)

    def _log(self, *args, **kwargs):
        raise NotImplementedError

    def _get_class_attributes(self) -> dict:
        """Attributes of the class that are configs of an operation

        Notes:
            This is used for logging the configs since they need to be manually 
            tuned or experimented with. I.e. for the same input, we might run
            this class (and the operation it implemented) multiple times with
            different configs to find the best config by human oracle verification.
            Hence, keeping the configs of each run even inside a single experiment is
            highly desired.

        Returns:
            dict: A dictionary of each parameter and its value
        """
        class_attributes: dict = dict(self.__dict__)
        # pop out logging instances and hidden attributes (ie. start with "_")
        d: dict = {}
        for k, v in class_attributes.items():
            if (not isinstance(v, logging.Logger)) and (not k.startswith('_')):
                d[k] = v

        class_attributes = d
        return class_attributes

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Detector(DetectorBase):
    """Base class for all detectors such as edge and line detectors
    """

    def __init__(self) -> None:
        super().__init__()

    def _log(self, *args, **kwargs):
        raise NotImplementedError

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class EdgeDetector(Detector):
    """Detect edges in an image
    """

    def __init__(self) -> None:
        super().__init__()

    def _log(self, *args, **kwargs):
        self.logger.info(
            f'{self.__class__.__name__} edge detection is performed'
            f' with kwargs: {kwargs}'
        )


class CannyEdgeDetector(EdgeDetector):
    """Detect edges in an image using Canny algorithm (cv2.Canny_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.6.0/da/d22/tutorial_py_canny.html>`_. 

    .. _cv2.Canny: https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    """

    def __init__(self,
                 threshold1: float,
                 threshold2: float,
                 aperture_size: int = 3,
                 L2_gradient: bool = False) -> None:
        """Initialize Canny edge detector

        Notes:
            For more info about the algorithm, see class docstring.

        Args:
            threshold1 (float): first threshold for the hysteresis procedure
            threshold2 (float): second threshold for the hysteresis procedure
            aperture_size (int): size of the Sobel kernel to be used
            L2_gradient (bool): use L2 norm rather than L1
        """
        super().__init__()

        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.L2_gradient = L2_gradient

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

    def __call__(self, image: np.ndarray,
                 plot: Optional[Path] = None,
                 *args, **kwargs) -> np.ndarray:
        # if kwargs provided, override class init
        self.threshold1 = kwargs.get('threshold1', self.threshold1)
        self.threshold2 = kwargs.get('threshold2', self.threshold2)
        self.aperture_size = kwargs.get('aperture_size', self.aperture_size)
        self.L2_gradient = kwargs.get('L2_gradient', self.L2_gradient)

        edges = self.detect(
            image,
            threshold1=self.threshold1,
            threshold2=self.threshold2,
            apertureSize=self.aperture_size,
            L2gradient=self.L2_gradient
        )

        # logging
        self._log(**self._get_class_attributes())
        if plot is not None:
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(edges, cmap='gray')
            plt.savefig(plot / 'canny_edge.png')
            plt.close(fig)

        return edges


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
        self.__horizontal_lines: List[np.ndarray] = []
        self.__vertical_lines: List[np.ndarray] = []
        self._image_shape: Optional[Tuple[int, int]] = None

    def _log(self, *args, **kwargs):
        self.logger.info(
            f'{self.__class__.__name__} line detection is performed'
            f' with kwargs: {kwargs}'
        )

    def _reset_lines(self):
        self.horizontal_lines: List[np.ndarray] = []
        self.vertical_lines: List[np.ndarray] = []

    @property
    def horizontal_lines(self) -> List[np.ndarray]:
        return self.__horizontal_lines

    @horizontal_lines.setter
    def horizontal_lines(self, lines: List[np.ndarray]):
        self.__horizontal_lines = lines

    @property
    def vertical_lines(self) -> List[np.ndarray]:
        return self.__vertical_lines

    @vertical_lines.setter
    def vertical_lines(self, lines: List[np.ndarray]):
        self.__vertical_lines = lines

    @staticmethod
    def _horizontal(line: np.ndarray) -> LineDirection:
        """Check if line is horizontal

        Args:
            line (:class:`numpy.ndarray`): line to check

        Returns:
            LineDirection: Direction of line
        """
        return line[1] == line[3]

    @staticmethod
    def _vertical(line: np.ndarray) -> LineDirection:
        """Check if line is vertical

        Args:
            line (:class:`numpy.ndarray`): line to check

        Returns:
            LineDirection: Direction of line
        """
        return line[0] == line[2]

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

    @staticmethod
    def _filter_overlapping_lines(
        lines: List[np.ndarray],
        sorting_index: int,
        separation: float = 5
    ) -> List[np.ndarray]:
        """Filter lines that are overlapping with each other

        Args:
            lines (List[:class:`numpy.ndarray`]): list of lines to filter
            sorting_index (int): index of line coordinate to sort by
            separation (float, optional): minimum separation between lines
                to be considered overlapping. Defaults to 5.

        Returns:
            List[:class:`numpy.ndarray`]:
            list of subset of filtered lines. i.e. :math:`text{out} \in text{lines}`

        TODO:

            * improve this method to consider all lines simultaneously. Currently,
                it only processed lines by the order they are sorted but if we use
                something like clustering, then we can find out the lines that are
                close to each other and then filter them out.
                This will also eliminate the need for sorting and tuning the ``separation``
                parameter for each image or perhaps, each group of lines.

        """
        filtered_lines: List[np.ndarray] = []

        lines = sorted(lines, key=lambda lines: lines[sorting_index])
        for i in range(len(lines)):
            current_line = lines[i]
            if (i > 0):
                previous_line = lines[i-1]
                if ((current_line[sorting_index] - previous_line[sorting_index]) > separation):
                    filtered_lines.append(current_line)
            else:
                filtered_lines.append(current_line)

        return filtered_lines

    def _get_border_lines(
        self,
        shape: Tuple[int, ...]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get lines that form the borders of an image

        Notes:
            This is in case where the detector algorithm does not detect edge
            lines that construct borders of an image. This is a common case when 
            there is not enough border; although we can solve this by adding border, 
            but if are going to add some code to add border, then why not manually give
            the lines that construct the borders?

        Args:
            shape (Tuple[int, ...]): shape (of image) to get border lines from

        Returns:
            :class:`numpy.ndarray`:
            A tuple of two lists of lines. The first list contains
                vertical lines and the second list contains horizontal lines.
        """
        left_border = np.array([2, 2, 2, shape[0]], dtype=np.int32)
        right_border = np.array(
            [shape[1], 2, shape[1], shape[0]], dtype=np.int32)
        top_border = np.array([2, 2, shape[1], 2], dtype=np.int32)
        bottom_border = np.array(
            [2, shape[0], shape[1], shape[0]], dtype=np.int32)
        vertical_lines = [left_border, right_border]
        horizontal_lines = [top_border, bottom_border]
        return vertical_lines, horizontal_lines

    def get_vertical_horizontal_lines(
        self,
        lines: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        # separate lines into vertical and horizontal
        # TODO: optimize this by boolean mask than for loop
        for line in lines:
            line = line.flatten()
            direction = self._find_line_direction(line)
            if direction == LineDirection.VERTICAL:
                vertical_lines.append(line)
            elif direction == LineDirection.HORIZONTAL:
                horizontal_lines.append(line)

        # add manually added borders
        vl_, hl_ = self._get_border_lines(self._image_shape)
        vertical_lines.extend(vl_)
        horizontal_lines.extend(hl_)
        # remove overlapping lines
        vertical_lines = self._filter_overlapping_lines(lines=vertical_lines,
                                                        sorting_index=0)
        horizontal_lines = self._filter_overlapping_lines(lines=horizontal_lines,
                                                          sorting_index=1)
        return vertical_lines, horizontal_lines

    def detect(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class ProbabilisticHoughLinesDetector(LineDetector):
    """Detect lines in an image using probabilistic Hough transform (cv2.HoughLinesP_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.6.0/d9/db0/tutorial_hough_lines.html>`_.

    .. _cv2.HoughLinesP: https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    """

    def __init__(self,
                 rho: float,
                 theta: float,
                 threshold: int,
                 min_line_length: int = 0,
                 max_line_gap: int = 0) -> None:
        """Initialize the detector

        Notes:
            For more info about the algorithm, see class docstring.

        Args:
            rho (float): distance resolution of the accumulator in pixels
            theta (float): angular resolution of the accumulator in radians
            threshold (int): accumulator threshold parameter. Only those lines 
                are returned that get enough votes ( > threshold )
            min_line_length (int, optional): minimum line length. Line segments
                shorter than that are rejected. Defaults to 0.
            max_line_gap (int, optional): maximum allowed gap between points on
                the same line to link them. Defaults to 0.
        """
        super().__init__()

        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

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

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self._image_shape = image.shape
        # if kwargs provided, override class attributes
        self.rho = kwargs.get('rho', self.rho)
        self.theta = kwargs.get('theta', self.theta)
        self.threshold = kwargs.get('threshold', self.threshold)
        self.min_line_length = kwargs.get(
            'min_line_length', self.min_line_length)
        self.max_line_gap = kwargs.get('max_line_gap', self.max_line_gap)

        self._log(**self._get_class_attributes())
        self._reset_lines()

        # get all lines
        lines = self.detect(image,
                            rho=self.rho,
                            theta=self.theta,
                            threshold=self.threshold,
                            minLineLength=self.min_line_length,
                            maxLineGap=self.max_line_gap)
        # separate lines into vertical and horizontal
        vertical_lines, horizontal_lines = self.get_vertical_horizontal_lines(
            lines)
        return vertical_lines, horizontal_lines


class NaiveHoughLinesDetector(LineDetector):
    """Detect lines in an image using naive Hough transform (cv2.HoughLines_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.6.0/d9/db0/tutorial_hough_lines.html>`_.

    .. _cv2.HoughLines: https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
    """

    def __init__(self,
                 rho: float,
                 theta: float,
                 threshold: int,
                 srn: float = 0.,
                 stn: float = 0.,
                 min_theta: float = 0.,
                 max_theta: float = np.pi) -> None:
        """Initialize the detector


        Notes:
            For more info about the algorithm, see class docstring.

        Args:
            rho (float): distance resolution of the accumulator in pixels
            theta (float): angular resolution of the accumulator in radians
            threshold (int): accumulator threshold parameter. Only those lines
                are returned that get enough votes ( > threshold )
            srn (float, optional): used to specify the maximum difference in
                radius between the circle centers, for the points to be
                considered as in the same circle. Defaults to 0.
            stn (float, optional): used to specify the maximum difference in
                theta between the circle centers, for the points to be
                considered as in the same circle. Defaults to 0.
            min_theta (float, optional): minimum angle to check for lines.
                Defaults to 0.
            max_theta (float, optional): maximum angle to check for lines.
                Defaults to :math:`\pi`.
        """
        super().__init__()

        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.srn = srn
        self.stn = stn
        self.min_theta = min_theta
        self.max_theta = max_theta

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

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self._image_shape = image.shape
        # if kwargs provided, override class attributes
        self.rho = kwargs.get('rho', self.rho)
        self.theta = kwargs.get('theta', self.theta)
        self.threshold = kwargs.get('threshold', self.threshold)
        self.srn = kwargs.get('srn', self.srn)
        self.stn = kwargs.get('stn', self.stn)
        self.min_theta = kwargs.get('min_theta', self.min_theta)
        self.max_theta = kwargs.get('max_theta', self.max_theta)

        self._log(**self._get_class_attributes())
        self._reset_lines()

        # get all lines
        lines = self.detect(image,
                            rho=self.rho,
                            theta=self.theta,
                            threshold=self.threshold,
                            srn=self.srn,
                            stn=self.stn,
                            min_theta=self.min_theta,
                            max_theta=self.max_theta)
        # separate lines into vertical and horizontal
        vertical_lines, horizontal_lines = self.get_vertical_horizontal_lines(
            lines)
        return vertical_lines, horizontal_lines


class ContourLinesDetector(LineDetector):
    """Detects lines via finding contours around solid areas

    For finding contours, cv2.findContours_ is being used.

    This method works best when the image is preprocessed and binary. For preprocessing,
    :mod:`ttocr.data.preprocessors` that has useful functions for this such as:

        * :class:`ttocr.data.preprocessors.Dilate`: for building a solid area of out a text
        * :class:`ttocr.data.preprocessors.OtsuThresholder`: for conversion to binary
        * :class:`ttocr.data.preprocessors.GaussianAdaptiveThresholder`: for conversion to binary
        * :class:`ttocr.data.preprocessors.GaussianImageSmoother`: for blurring the image

    Notes:
        For more info about the algorithm, see https://docs.opencv.org/4.6.0/d4/d73/tutorial_py_contours_begin.html

    .. _cv2.findContours: https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    """

    def __init__(self,
                 min_solid_height_limit: Optional[int] = None,
                 max_solid_height_limit: Optional[int] = None,
                 cell_threshold: Optional[int] = None,
                 min_columns: Optional[int] = None,
                 ) -> None:
        """Initialize the detector

        Args:
            min_solid_height_limit (int, optional): minimum height of solid area to
                detect, i.e. if detected contour's height is smaller than this,
                it will be ignored. Recommended value is ``6``. *Tuning this value
                in an human-in-the-loop way is recommended.*
            max_solid_height_limit (int, optional): maximum height of solid area to
                detect, i.e. if detected contour's height is larger than this,
                it will be ignored. Recommended value is ``40``. *Tuning this value
                in an human-in-the-loop way is recommended.*
            cell_threshold (int, optional): bin sizes for clustering detected contours
                into rows and columns. Recommended value is ``10``.
            min_columns (int, optional): minimum number of columns to detect.
                It works much better when input image has a single column. Hence,
                recommended value is ``1``.
        """
        super().__init__()

        self.min_solid_height_limit = min_solid_height_limit
        self.max_solid_height_limit = max_solid_height_limit
        self.cell_threshold = cell_threshold
        self.min_columns = min_columns

    @staticmethod
    def _find_solid_boxes(image: np.ndarray,
                          min_solid_height_limit: int,
                          max_solid_height_limit: int) -> List[np.ndarray]:
        """Find solid boxes in the image via finding rectangular (box) contours around them

        Args:
            image (:class:`numpy.ndarray`): image to detect contours
            min_solid_height_limit (int): minimum height of solid area (see class docstring)
            max_solid_height_limit (int): maximum height of solid area (see class docstring)

        Returns:
            List[:class:`numpy.ndarray`]: list of detected contours
        """
        # TODO: check return type
        # looking for the solid spots contours
        contours, hierarchy = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # getting the solids bounding boxes based on thesolidt size assumptions
        boxes: List = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            h = box[3]
            if min_solid_height_limit < h < max_solid_height_limit:
                boxes.append(box)
        return boxes

    @staticmethod
    def _detect_table_of_boxes(boxes: List[np.ndarray],
                               cell_threshold: int,
                               min_columns: int
                               ) -> Union[List[List[np.ndarray]], List, None]:
        """Detect table out of given rectangular boxes

        Args:
            boxes (List[:class:`numpy.ndarray`]): list of rectangular boxes found by
                :func:`_find_solid_boxes`
            cell_threshold (int): bin sizes for clustering detected contours
                into rows and columns. See class docstring.
            min_columns (int): minimum number of columns to detect. 
                See class docstring.

        Returns:
            Union[List[List[:class:`numpy.ndarray`]], List, None]:
                table (rows and columns) of detected boxes. It could be empty list
                or None which means that no table was detected.
        """
        # TODO: check return type
        rows: dict = {}
        cols: dict = {}

        # clustering the bounding boxes by their positions
        for box in boxes:
            (x, y, w, h) = box
            col_key = x // cell_threshold
            row_key = y // cell_threshold
            cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
            rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

        # filtering out the clusters having less than `min_columns` cols
        table_cells = list(
            filter(lambda r: len(r) >= min_columns, rows.values()))
        # sorting the row cells by x coord
        table_cells = [list(sorted(tb)) for tb in table_cells]
        # sorting rows by the y coord
        table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

        return table_cells

    def _build_lines(
        self,
        table_cells: Optional[List[List[np.ndarray]]]
    ) -> Tuple[List, List]:
        """Build horizontal and vertical lines out of given table cells

        Args:
            table_cells (Optional[List[List[np.ndarray]]]): a table as list of lists of boxes.
                See :func:`_detect_table_of_boxes` for more info.

        Returns:
            Tuple[List, List]: vertical and horizontal lines
        """
        if table_cells is None or len(table_cells) <= 0:
            return [], []

        # find largest row (start=min_x, end=max_x)
        max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
        max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]
        min_x = max(table_cells, key=lambda b: b[-1][2])[-1][0]

        # find largest column (..., end=max_y)
        max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
        max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

        hor_lines: Union[List[Tuple[int, ...]], List] = []
        ver_lines: Union[List[Tuple[int, ...]], List] = []

        for box in table_cells:
            x = box[0][0]
            y = box[0][1]
            hor_lines.append((min_x, y, max_x, y))

        for box in table_cells[0]:
            x = box[0]
            y = box[1]
            ver_lines.append((min_x, y, min_x, max_y))

        (x, y, w, h) = table_cells[0][-1]
        ver_lines.append((max_x, y, max_x, max_y))
        (x, y, w, h) = table_cells[0][0]
        hor_lines.append((min_x, max_y, max_x, max_y))

        # remove overlapping lines
        ver_lines = self._filter_overlapping_lines(lines=ver_lines,
                                                   sorting_index=0)
        hor_lines = self._filter_overlapping_lines(lines=hor_lines,
                                                   sorting_index=1)
        return ver_lines, hor_lines

    def __call__(self, image: np.ndarray,
                 plot: Optional[Path] = None,
                 *args, **kwargs) -> Tuple[List, List]:
        self._reset_lines()
        # if kwargs are provided, override the class attributes
        self.min_solid_height_limit = kwargs.get('min_solid_height_limit',
                                                 self.min_solid_height_limit)
        self.max_solid_height_limit = kwargs.get('max_solid_height_limit',
                                                 self.max_solid_height_limit)
        self.cell_threshold = kwargs.get('cell_threshold', self.cell_threshold)
        self.min_columns = kwargs.get('min_columns', self.min_columns)

        solid_boxes = self._find_solid_boxes(
            image=image,
            min_solid_height_limit=self.min_solid_height_limit,
            max_solid_height_limit=self.max_solid_height_limit
        )
        cells = self._detect_table_of_boxes(
            boxes=solid_boxes,
            cell_threshold=self.cell_threshold,
            min_columns=self.min_columns
        )
        vertical_lines, horizontal_lines = self._build_lines(cells)

        # logging
        self._log(**self._get_class_attributes())
        if plot is not None:
            fig = plt.figure(figsize=(12, 12))
            __all_lines = vertical_lines + horizontal_lines
            vis = visualizers.draw_lines(image=image,
                                         lines=__all_lines,
                                         color=(0, 0, 255),
                                         copy=True)
            plt.imshow(vis, cmap='gray')
            plt.savefig(plot / 'contour_lines.png')
            plt.close(fig)

        return vertical_lines, horizontal_lines


class OCR(Detector):
    """Detect text in an image using Optical Character Recognition (OCR)
    """

    def __init__(self) -> None:
        super().__init__()

    def _log(self, *args, **kwargs):
        self.logger.info(
            f'{self.__class__.__name__} OCR is performed'
            f' with kwargs: {kwargs}'
        )


class TesseractOCR(OCR):
    """Does OCR using Google's Tesseract OCR engine

    For more info about Tesseract, see https://tesseract-ocr.github.io/.
    """

    def __init__(self,
                 l: str = 'eng+fas',
                 dpi: int = 100,
                 psm: int = 6,
                 oem: int = 3,
                 *args
                 ) -> None:
        """Initialize the detector

        Notes:
            For more info about Tesseract, see class docstring.

        Args:
            l (str, optional): language to use. Defaults to 'eng+fas'.
                You can add more languages by ``'lang1+lang2+...+langN'``
            dpi (int, optional): dpi to use. Defaults to 100.
            psm (int, optional): page segmentation mode. Defaults to 6.
            oem (int, optional): OCR engine mode. Defaults to 3.
            *args: arguments for ``tesseract``
        """
        super().__init__()

        self.l = l
        self.dpi = dpi
        self.psm = psm
        self.oem = oem
        self.args = args

    def __kwargs_to_string(self, *args, **kwargs) -> str:
        """Convert keyword arguments to string for Tesseract command line
        """

        config: str = ''
        for key, value in kwargs.items():
            if len(key) > 1:     # e.g. --dpi 100
                config += f'--{key} {value} '
            elif len(key) == 1:  # e.g -l fas
                config += f'-{key} {value} '
            # args for CONFIGFILE
            else:
                pass
        for arg in args:         # e.g. hocr
            config += f'{arg} '
        return config

    def detect(self, image: np.ndarray, config: str, *args, **kwargs) -> np.ndarray:
        """Detect text in an image using Google's Tesseract OCR engine

        Args:
            image (:class:`numpy.ndarray`): image to detect text in
            config (str): configuration string for Tesseract
            kwargs: keyword arguments for ``tesseract`` CLI command.
                Most important one are:

                    - ``-l LANG``: language to use for OCR
                    - ``--dpi N``: DPI to use for OCR
                    - ``--oem N``: OCR engine mode
                    - ``--psm N``: Page segmentation mode. Suggested values
                        are 6 an 3. Defaults to 3.

                An example:
                    ``tesseract --oem 3 --psm 6 -l eng+fas --dpi 100 image.png output hocr pdf txt``           

        Returns:
            :class:`numpy.ndarray`: image with detected text
        """
        # pytesseract requires kwargs to be string as CLI command
        config = self.config
        config_: str = ''
        if kwargs is not None:
            config_ = self.__kwargs_to_string(*args, **kwargs)
            config_ = config_.strip()
        # append to self.config
        config = config + config_
        config = config.strip()
        text = pytesseract.image_to_string(image, config=config)
        return text

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Detect text in an image using Google's Tesseract OCR engine

        Args:
            image (:class:`numpy.ndarray`): image to detect text in
            kwargs: keyword arguments for ``tesseract`` CLI command.
                Most important one are:

                    - ``-l LANG``: language to use for OCR
                    - ``--dpi N``: DPI to use for OCR
                    - ``--oem N``: OCR engine mode
                    - ``--psm N``: Page segmentation mode. Suggested values
                        are 6 an 3. Defaults to 3.

                An example:
                    ``tesseract --oem 3 --psm 6 -l eng+fas --dpi 100 image.png output hocr pdf txt``


        Returns:
            :class:`numpy.ndarray`: image with detected text
        """
        # if kwargs provided, override class attributes
        self.l = kwargs.get('l', self.l)
        self.dpi = kwargs.get('dpi', self.dpi)
        self.psm = kwargs.get('psm', self.psm)
        self.oem = kwargs.get('oem', self.oem)
        self.args = kwargs.get('args', self.args)
        self._log(**self._get_class_attributes())

        self.config = self.__kwargs_to_string(
            l=self.l,
            dpi=self.dpi,
            psm=self.psm,
            oem=self.oem,
            *self.args
        )

        return self.detect(image, self.config, *args, **kwargs)


class TableCellDetector(LineDetector):
    """Detects cells of a table via detected vertical and horizontal lines

    Notes:
        For more info about obtaining lines, see :class:`LineDetector`
        and other classes that subclass it, e.g. :class:`ProbabilisticHoughLinesDetector`.

    TODO:
        Separate table cell **detection** from table cell **OCR**. I.e. Detecting table cells
        is an extension of detecting lines (forming a table cell from lines) and could be used
        anywhere. But OCR is an absolute different task that could be done separately too.
        So, now generate a 2d list of cell images via `TableCellDetector` and then create
        a new class e.g. `TableCellOCR` that does OCR on each cell image.

    """

    def __init__(self,
                 ocr: Optional[OCR] = None,
                 roi_offset: Optional[int] = None) -> None:
        """

        Args:
            ocr (OCR): OCR instance to use for text detection. Has
                to be an instance of :class:`OCR`. E.g. see :class:`TesseractOCR`.
                Defaults to :class:`TesseractOCR`.
            roi_offset (int, optional): offset of the ROI. If :class:`ContourLinesDetector`
                is used for line detection, recommended to be ``0``, otherwise, ``4``.
        """
        super().__init__()

        self.__ocr = ocr
        self.roi_offset = roi_offset
        self.__ocred_cells: np.ndarray = None

    @property
    def _num_rows(self) -> int:
        """Number of rows in table
        """
        nr = len(self.horizontal_lines) - 1
        # log warning if number of rows is zero
        if nr == 0:
            self.logger.warning('Number of rows is zero!')
        return nr

    @property
    def _num_columns(self) -> int:
        """Number of columns in table
        """
        nc = len(self.vertical_lines) - 1
        # log warning if number of columns is zero
        if nc == 0:
            self.logger.warning('Number of columns is zero!')
        return nc

    def _log(self):
        self.logger.info('Using TableCellDetector')
        self.logger.info(
            f'Possible table size: {self._num_rows} x {self._num_columns}')

    @property
    def ocred_cells(self) -> np.ndarray:
        """A 2D array of detected cells
        """
        return self.__ocred_cells

    @ocred_cells.setter
    def ocred_cells(self, value: np.ndarray) -> None:
        self.__ocred_cells = value

    @staticmethod
    def _crop_image(image: np.ndarray,
                    x: int, y: int, w: int, h: int) -> np.ndarray:
        """Crops an image to a specified region

        Args:
            image (:class:`numpy.ndarray`): image to crop
            x (int): x coordinate of top left corner of crop
            y (int): y coordinate of top left corner of crop
            w (int): width of crop
            h (int): height of crop

        Returns:
            :class:`numpy.ndarray`: cropped image with ``shape=(h, w)``
        """
        return image[y:y+h, x:x+w]

    def _extract_region_of_interest(self, image: np.ndarray,
                                    horizontal_lines: List[np.ndarray],
                                    vertical_lines: List[np.ndarray],
                                    row_index: int,
                                    col_index: int,
                                    offset: int = 4,
                                    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Extract ROI as cells bounded by horizontal and vertical lines

        Args:
            image (:class:`numpy.ndarray`): image to extract ROI from
            horizontal_lines (List[:class:`numpy.ndarray`]): list of horizontal lines
            vertical_lines (List[:class:`numpy.ndarray`]): list of vertical lines
            row_index (int): index of row to extract columns from
            col_index (int): index of column to extract
            offset (int, optional): offset from line to extract ROI. Defaults to 4.

        Returns:
            :class:`numpy.ndarray`: extracted ROI
        """
        left_line_index = col_index
        right_line_index = col_index + 1
        top_line_index = row_index
        bottom_line_index = row_index + 1

        x1 = vertical_lines[left_line_index][2] + offset
        y1 = horizontal_lines[top_line_index][3] + offset
        x2 = vertical_lines[right_line_index][2] - offset
        y2 = horizontal_lines[bottom_line_index][3] - offset

        # relax offsets when x1=x2 or y1=y2
        if x1 == x2:
            x1 = x1 - (offset // 2)
            x2 = x2 + (offset // 2)
        if y1 == y2:
            y1 = y1 - (offset // 2)
            y2 = y2 + (offset // 2)

        w = x2 - x1
        h = y2 - y1

        # cropped ROI
        roi = self._crop_image(image, x1, y1, w, h)
        return roi, (x1, y1, w, h)

    def __call__(
        self,
        image: np.ndarray,
        plot: Union[Path, str, None] = None,
        *args, **kwargs
    ) -> Union[Union[List[List[str]], np.ndarray], List[List[str]]]:
        """

        Args:
            image (:class:`numpy.ndarray`): image to detect cells
            plot (str, optional): Path to plot detected cells. If None, then
                no plot is created. Defaults to None.

        Returns:
                Union[Union[List[List[str]], np.ndarray], List[List[str]]]:
                When ``plot`` is None, only returns `List[List[str]]` 
                as the OCR result, otherwise a tuple of 
                OCRed text and annotated image.

        """
        # if kwargs provided, override class attributes
        self.roi_offset = kwargs.get('roi_offset', self.roi_offset)
        self._log()

        # all detected cells
        # TODO: use np.empty instead for optimization
        # ocred_cells: np.ndarray = np.empty((self._num_rows,
        #                                     self._num_columns),
        #                                     dtype=object)

        # all ocred cells
        ocred_cells: List[List[str]] = []

        if plot is not None:
            # count number of cells for subplot layout
            counter = 0
            fig = plt.figure(figsize=(12, 12))
            fig.set_facecolor('k')

        # iterate through all cells
        for i in range(self._num_rows):
            ocred_row: List[str] = []
            for j in range(self._num_columns):
                roi, _ = self._extract_region_of_interest(
                    image=image,
                    horizontal_lines=self.horizontal_lines,
                    vertical_lines=self.vertical_lines,
                    row_index=i,
                    col_index=j,
                    offset=self.roi_offset,
                )

                # skip if ROI is empty
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    self.logger.warning(f'ROI is empty at {i}, {j}, skipped.')
                    continue

                # detect text via OCR
                text = self.__ocr(roi)
                ocred_row.append(text)
                self.logger.debug(f'Cell {i}x{j} has text: "{text}"')

                # plot detected cells as table
                if plot is not None:
                    counter += 1
                    ax = fig.add_subplot(
                        self._num_rows, self._num_columns, counter)
                    plt.subplots_adjust(hspace=2)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.imshow(roi)
                    plt.title(text,
                              fontdict={
                                  'fontsize': 8,
                                  'verticalalignment': 'center'
                              },
                              color='white')
            # save ocred row to list of ocred cells
            ocred_cells.append(ocred_row)

        self.ocred_cells = ocred_cells

        # save plot
        if plot is not None:
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(),
                                            dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )

            plt.savefig(plot / 'table-ocr.png', facecolor='k')
            plt.close()

            # for debug, return texts and image with annotations
            return self.ocred_cells, image_from_plot
        return self.ocred_cells
