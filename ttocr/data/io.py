__all__ = [
    'ImageReader', 'CV2ImageReader', 
]

# core
import numpy as np
import cv2
# helpers
import logging


# setup logger
logger = logging.getLogger(__name__)


class ImageReader:
    """Base class for image type readers, e.g. numpy and cv2

    User should subclass this class and implement :meth:`read` method.
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(logger.name+'.'+self.__class__.__name__)
        pass
    
    def __log(self):
        raise NotImplementedError

    def __call__(self, image_path: str, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
class CV2ImageReader(ImageReader):
    """Read image from file path and return numpy array via ``OpenCV``
    
    """
    def __init__(self) -> None:
        super().__init__()
    
    def __log(self, path: str) -> None:
        self.logger.info(f'Image from {path} is read')
    
    def __call__(self, image_path: str, *args, **kwargs) -> np.ndarray:
        self.__log(image_path)
        img = cv2.imread(cv2.samples.findFile(image_path), *args, **kwargs)
        return img

