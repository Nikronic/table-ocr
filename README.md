# tour-table-ocr

OCRing the tables of tours to be able to cast them to any format, structure or template.

## Install
Install tesseract system level and remove any conda package installation of it. I.e. when you make calls to tesseract,
it has to run the system wide tesseract than conda one. 

## Branches
1. ttocr-ml
2. ttocr-dl

### ttocr-ml
The branch that is main focused on machine learning and classic vision methods for parsing images of text data. For instance, using OpenCV line detection
algorithms for detecting tables, rows, columns and so on.

### ttocr-dl
The branch that is main focused on deep learning methods for parsing the images of text data. For instance, using
[MS Layout Parser](https://github.com/Layout-Parser/layout-parser) for extracting table or table structure.

Note that OCRing itself is never considered a problem to be solved and I assume, always we can OCR a "clean" image. ("clean" image means an image that does not contain any noise or other issues that OCR can't see the text.)

## Remarks
1. **Code Quality**: Given the fact that not only I wasn't rewarded for writing clean/modular/future-proof code and documentation, I was punished for not delivering
a usable product sooner (although when I started the job, I clearly stated that I plan to spend time on clean code and the work we have needs huge manual labeling,
preprocessing and so on and please let me know if you have any expectation and the answer I got was like "We don't have any expectations, we believe all you say.").
Hence, I am no longer writing a clean code and practically will throw everything into a single notebook or `main.py` file!
