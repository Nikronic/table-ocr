import requests
import json
from pathlib import Path

url = 'http://127.0.0.1:8000/flag'
example_image = Path('test_sample/s.png')
files = {
    # 'filename': example_image.as_posix(),
    'file': (example_image.as_posix(),
             open(example_image.as_posix(), 'rb'),
             'image/png')
    # 'content_type': 'image/png',
    }
data = {'conf': json.dumps(
    {
        "mode": True,
        "canny_threshold1": 50,
        "canny_threshold2": 200,
        "canny_aperture_size": 3,
        "canny_L2_gradient": False,
        "hough_min_line_length": 50,
        "hough_max_line_gap": 10,
        "smooth_kernel_size": 3,
        "thresh_block_size": 11,
        "thresh_c": 5,
        "dilate_morph_size": 3,
        "dilation_iterations": 3,
        "contour_line_cell_threshold": 10,
        "contour_min_solid_height_limit": 6,
        "contour_max_solid_height_limit": 40,
        "roi_offset": 0,
        "ocr_lang": "eng",
        "ocr_dpi": 150,
        "ocr_psm": 6,
        "ocr_oem": 1
        }   
    )
}

resp = requests.post(url=url, data=data, files=files) 
print(resp.json())

