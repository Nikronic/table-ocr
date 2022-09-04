
__all__ = [
    'create_engine'
    'Base', 'TTOCRMLConfigs'
]

# core
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
# helper
import logging

# configure logger
logger = logging.getLogger(__name__)


# create a DeclarativeMeta instance
Base = declarative_base()

# define class inheriting from Base
class TTOCRMLConfigs(Base):
    # DATABASE CONFIG
    __tablename__ = 'ttocr_ml_configs'
    idx = Column(Integer, primary_key=True)

    # OUR OCR CONFIG
    mode = Column(Boolean)

    # preprocessing for ML_TABLE
    canny_threshold1 = Column(Float)
    canny_threshold2 = Column(Float)
    canny_aperture_size = Column(Integer)
    canny_L2_gradient = Column(Boolean)

    hough_min_line_length = Column(Integer)
    hough_max_line_gap = Column(Integer)

    # preprocessing for ML_SINGLE_COLUMN_TABLE
    smooth_kernel_size = Column(Integer)

    thresh_block_size = Column(Integer)
    thresh_c = Column(Integer)

    dilate_morph_size = Column(Integer)
    dilation_iterations = Column(Integer)

    contour_line_cell_threshold = Column(Integer)
    contour_min_solid_height_limit = Column(Integer)
    contour_max_solid_height_limit = Column(Integer)

    roi_offset = Column(Integer)

    # common
    ocr_lang = Column(String(50))
    ocr_dpi = Column(Integer)
    ocr_psm = Column(Integer)
    ocr_oem = Column(Integer)
