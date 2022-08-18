__all__ = [
    'draw_lines'
]

# core
import matplotlib.pyplot as plt
import numpy as np
import cv2
# helpers
from typing import List, Optional, Tuple


def draw_lines(image: np.ndarray,
               lines: List[np.ndarray],
               color: Tuple = (0, 0, 255),
               thickness: int = 3,
               name: Optional[str] = None,
               copy: bool = True,
               ) -> np.ndarray:
    """Draw lines on image using OpenCV's line drawing function (cv2.line_)

    Args:
        image (:class:`numpy.ndarray`): Image to draw lines on
        lines (List[:class:`numpy.ndarray`]): List of lines to draw.
            Each item (line) is a numpy array of shape (4,) where
            the first two elements are the x and y coordinates of the
            start point and the last two are the x and y coordinates
            of the end point.
        color (tuple): Color of lines to draw as a tuple
            of ``(R, G ,B)``. Default is red.
        thickness (int): Thickness of lines to draw. Default is 3.
        name (str, optional): If str, draw the :math:`i`th line with
            name ``i{str}`` for lines, otherwise, nothing will be drawn.
            Default is None. 
        copy (bool): If True, return a copy of the image with lines drawn,
            otherwise original image will be modified. Default is True.
    Returns:
        :class:`numpy.ndarray`: Image with lines drawn on it

    .. _cv2.line: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
    """

    if copy:
        image = np.copy(image)

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        if name is not None:
            cv2.putText(image, str(i) + name,
                        (line[0] + 5, line[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image
