# core
import numpy as np
import cv2
# ours: data
from ttocr.data import io
from ttocr.data import preprocessors
# ours: detection
from ttocr.detection import detectors
# ours: utils
from ttocr.version import VERSION as TTOCR_VERSION
# devops
import mlflow
# API
from starlette import applications
from starlette import templating
from starlette import staticfiles
from starlette import routing
import pydantic
import uvicorn
import fastapi
import base64
# helpers
from typing import Any, Optional
from time import perf_counter
from pathlib import Path
import logging
import shutil
import sys
import os


# globals
SEED = 8569
VERBOSE = logging.DEBUG
DEVICE = 'cuda'

# configure logging
logger = logging.getLogger(__name__)
logger.setLevel(VERBOSE)
logger_formatter = logging.Formatter(
    "[%(name)s: %(asctime)s] {%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
)

# setup fastapi app
app = fastapi.FastAPI()
templates =  templating.Jinja2Templates(directory='templates')
web_app = applications.Starlette(
    routes=[
        routing.Mount("/static", app=staticfiles.StaticFiles(directory="static"),
                      name="static"),
    ],
)
app.mount("/web_app", web_app)

class OCRImage(pydantic.BaseModel):
    name: Optional[str] = None
    image: str
    doc_type: int = 0

def bytes_to_ndarray(data: bytes) -> np.ndarray:
    # https://stackoverflow.com/a/52495126/873282
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def ndarray_to_bytes(data: np.ndarray, image_type=".jpg"):
    return cv2.imencode(image_type, data)[1].tobytes()

def base64str_to_ndarray(data: str) -> np.ndarray:
    return bytes_to_ndarray(base64.b64decode(data))

def ndarray_to_base64str(data: np.ndarray) -> str:
    # https://stackoverflow.com/a/57830004
    return base64.b64encode(ndarray_to_bytes(data)).decode()
    
def get(dict_obj: dict, key: Any, default_value: Optional[Any] = None):
    """
    Like dict.get() but returns default value if the value is None or key is missing.
    """
    value = dict_obj.get(key, default_value)
    if value is None:
        value = default_value
    return 

@app.websocket("/ttocr/{mode}")
async def ocr_doc(websocket: fastapi.WebSocket, mode: str):
    if mode not in ('real', 'debug_upload'):
        raise ValueError(
            f'Expected parameter `mode` to be one of' 
            f' (`real`, `debug_upload`), saw `{mode}`'
        )
    await websocket.accept()
    try:
        while True:
            item = await websocket.receive_json(mode='text')
            tic = perf_counter()
            item = OCRImage(**item)
            img = base64str_to_ndarray(item.image)
            # copy image for visualization (draw on image)
            copy_image = np.copy(img)

            # convert color to gray
            color_converter = preprocessors.CV2ImageColorConverter()
            gray_img = color_converter(image=img,
                mode=preprocessors.CV2ImageColorConverterModes.BGR2GRAY)

            # detect canny edges
            canny_detector = detectors.CannyEdgeDetector()
            canny_edges = canny_detector(image=gray_img,
                                            threshold1=50,
                                            threshold2=200)

            # detect lines
            line_detector = detectors.ProbabilisticHoughLinesDetector()
            lines = line_detector(image=canny_edges,
                                    rho=1,
                                    theta=np.pi / 180,
                                    threshold=100,
                                    minLineLength=350,
                                    maxLineGap=18)
            
            # define ocr engine
            ocr_engine = detectors.TesseractOCR()
            table_cell_ocr = detectors.TableCellDetector(ocr=ocr_engine)
            table_cell_ocr.vertical_lines = lines[0]
            table_cell_ocr.horizontal_lines = lines[1]
            res = table_cell_ocr(image=gray_img,
                                 plot=Path('temp'))
            output = {
                "det_success": 1,
                "doc": ndarray_to_base64str(img),
                "doc_vis": ndarray_to_base64str(res[1]),
                "ocred_cells": res[0],
            }

            elapsed_time = perf_counter() - tic
            logger.info(f'Time taken: {elapsed_time:.6f} sec')
            await websocket.send_json(output, mode='text')
    except fastapi.WebSocketDisconnect:
        logger.info(f'"WebSocket {websocket.url.path}" [disconnected]')
    except Exception as e:
        logger.exception(e)
        raise e

@app.get('/', response_class=fastapi.responses.HTMLResponse)
@app.get('/upload', response_class=fastapi.responses.HTMLResponse)
async def upload_page(request: fastapi.Request):
    return templates.TemplateResponse('upload_ws.html', {'request': request})


if __name__ == '__main__':
    host = str(os.getenv("WEBSOCKET_HOST", "0.0.0.0"))
    port = int(os.getenv("WEBSOCKET_PORT", 8000))
    log_level = str(os.getenv("WEBSOCKET_LOG_LEVEL", "debug"))
    uvicorn.run("main:app", host=host, port=port, log_level=log_level)
