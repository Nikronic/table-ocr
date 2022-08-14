__all__ = [
    # abstract classes
    'Detector', 'EdgeDetector', 'LineDetector', 'OCR',
    # concrete classes
    'CannyEdgeDetector', 'ProbabilisticHoughLinesDetector', 'NaiveHoughLinesDetector',
    'TesseractOCR', 'TableCellDetector'
]

# core
import pytesseract
import numpy as np
import cv2
# helpers
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import logging


# setup logger
logger = logging.getLogger(__name__)


class Detector:
    """Base class for all detectors such as edge and line detectors

    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            logger.name+'.'+self.__class__.__name__)

    def _log(self):
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

    def _log(self):
        self.logger.info(
            f'{self.__class__.__name__} edge detection is performed')


class CannyEdgeDetector(EdgeDetector):
    """Detect edges in an image using Canny algorithm (cv2.Canny_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.6.0/da/d22/tutorial_py_canny.html>`_. 
    
    .. _cv2.Canny: https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
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
    
    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self._log()
        return self.detect(image, *args, **kwargs)


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

    def _log(self):
        self.logger.info(
            f'{self.__class__.__name__} line detection is performed')

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
    def _filter_overlapping_lines(lines: List[np.ndarray],
                                  sorting_index: int,
                                  separation: float = 5) -> List[np.ndarray]:
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
            if(i > 0):
                previous_line = lines[i-1]
                if ((current_line[sorting_index] - previous_line[sorting_index]) > separation):
                    filtered_lines.append(current_line)
            else:
                filtered_lines.append(current_line)
                    
        return filtered_lines
    
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
        # separate lines into vertical and horizontal
        # TODO: optimize this by boolean mask than for loop
        for line in lines:
            line = line.flatten()
            direction = self._find_line_direction(line)
            if direction == LineDirection.VERTICAL:
                vertical_lines.append(line)
            elif direction == LineDirection.HORIZONTAL:
                horizontal_lines.append(line)
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
    
    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self._log()
        self._reset_lines()

        # get all lines
        lines = self.detect(image, *args, **kwargs)
        # separate lines into vertical and horizontal
        vertical_lines, horizontal_lines = self.get_vertical_horizontal_lines(lines)
        return vertical_lines, horizontal_lines


class NaiveHoughLinesDetector(LineDetector):
    """Detect lines in an image using naive Hough transform (cv2.HoughLines_)

    Notes:
        For more info about the algorithm, see
        `here <https://docs.opencv.org/4.6.0/d9/db0/tutorial_hough_lines.html>`_.

    .. _cv2.HoughLines: https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
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
    
    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        self._log()
        self._reset_lines()

        # get all lines
        lines = self.detect(image, *args, **kwargs)
        # separate lines into vertical and horizontal
        vertical_lines, horizontal_lines = self.get_vertical_horizontal_lines(lines)
        return vertical_lines, horizontal_lines


class OCR(Detector):
    """Detect text in an image using Optical Character Recognition (OCR)
    """

    def __init__(self) -> None:
        super().__init__()


class TesseractOCR(OCR):
    """Does OCR using Google's Tesseract OCR engine

    For more info about Tesseract, see https://tesseract-ocr.github.io/.
    """

    def __init__(self) -> None:
        super().__init__()
        self._log()

    def _log(self):
        self.logger.info("Using Tesseract OCR")

    def detect(self, image: np.ndarray, **kwargs) -> np.ndarray:
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
        # pytesseract requires kwargs to be string as CLI command
        config: str = ''
        if kwargs is not None:
            for key, value in kwargs.items():
                if len(key) > 1:     # e.g. --dpi 100
                    config += f'--{key} {value} '
                elif len(key) == 1:  # e.g -l fas
                    config += f'-{key} {value} '
                # args for CONFIGFILE
                else:                # e.g. hocr
                    config += f'{value} '
        else:
            config = '--dpi 100 -l eng+fas --oem 3 --psm 6'
        config = config.strip()
        text = pytesseract.image_to_string(image, config=config)
        return text

    def __call__(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.detect(image, *args, **kwargs)


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

    def __init__(self, ocr: OCR = TesseractOCR) -> None:
        """

        Args:
            ocr (OCR): OCR instance to use for text detection. Has
                to be an instance of :class:`OCR`. E.g. see :class:`TesseractOCR`.
                Defaults to :class:`TesseractOCR`.
        """
        super().__init__()

        self.__ocr = ocr
        self.__ocred_cells: np.ndarray = None
    
    @property
    def _num_rows(self) -> int:
        """Number of rows in table
        """
        return len(self.horizontal_lines) - 1
    
    @property
    def _num_columns(self) -> int:
        """Number of columns in table
        """
        return len(self.vertical_lines) - 1

    def _log(self):
        self.logger.info('Using TableCellDetector')
        self.logger.info(f'Possible table size: {self._num_rows} x {self._num_columns}')

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
                    x: int, y: int, w: int, h:int) -> np.ndarray:
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
        w = x2 - x1
        h = y2 - y1

        # cropped ROI
        roi = self._crop_image(image, x1, y1, w, h)
        return roi, (x1, y1, w, h)


    def __call__(self, image: np.ndarray,
                 plot: Union[Path, str, None] = None,
                 *args, **kwargs) -> np.ndarray:
        """

        Args:
            image (:class:`numpy.ndarray`): image to detect cells
            plot (str, optional): Path to plot detected cells. If None, then
                no plot is created. Defaults to None.
                Note that this is for debugging purposes only hence hidden from class definition.

        """
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
                    offset=4,
                )

                # detect text via OCR
                text = self.__ocr(roi)
                ocred_row.append(text)
        
                # plot detected cells as table
                if plot is not None:
                    counter += 1
                    ax = fig.add_subplot(self._num_rows, self._num_columns, counter)
                    plt.subplots_adjust(hspace=2)
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.imshow(roi)
                    plt.title(text,
                              fontdict={
                                'fontsize':8,
                                'verticalalignment': 'center'
                              },
                              color='white')
            # save ocred row to list of ocred cells
            ocred_cells.append(ocred_row)

        self.ocred_cells = ocred_cells

        # save plot
        if plot is not None:
            plt.savefig(plot / 'table-ocr.png', facecolor='k')
            plt.close()
        return self.ocred_cells
