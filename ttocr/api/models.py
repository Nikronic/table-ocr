# core
import pydantic
import json
# helpers
from typing import List, Any


class BaseModel(pydantic.BaseModel):
    """
    Reference:
        * https://stackoverflow.com/a/70640522/18971263
    """
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

    @classmethod
    def __get_validators__(cls):
        yield cls._validate_from_json_string

    @classmethod
    def _validate_from_json_string(cls, value):
        if isinstance(value, str):
            return cls.validate(json.loads(value.encode()))
        return cls.validate(value)


class PredictionResponse(BaseModel):
    ocr_result: List[List[str]]


class Payload(BaseModel):
    name: str

    mode: bool = True

    # preprocessing for ML_TABLE
    canny_threshold1: float = 50
    canny_threshold2: float = 200
    canny_aperture_size: int = 3
    canny_L2_gradient: bool = False

    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # preprocessing for ML_SINGLE_COLUMN_TABLE
    smooth_kernel_size: int = 3

    thresh_block_size: int = 11
    thresh_c: int = 5

    dilate_morph_size: int = 3
    dilation_iterations: int = 3

    contour_line_cell_threshold: int = 10
    contour_min_solid_height_limit: int = 6
    contour_max_solid_height_limit: int = 40

    roi_offset: int = 0

    # common
    ocr_lang: str = 'eng'
    ocr_dpi: int = 150
    ocr_psm: int = 6
    ocr_oem: int = 1

    class Config:
        orm_mode = True
