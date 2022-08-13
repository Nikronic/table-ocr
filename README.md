# tour-table-ocr
OCRing the tables of tours to be able to cast them to any format, structure or template.

## Install
Install tesseract system level and remove any conda package installation of it. I.e. when you make calls to tesseract,
it has to run the system wide tesseract than conda one. 

## Branches
1. ttocr-ml
2. ttocr-dl

### ttocr-ml
> Use virtual env `ttocrml39`

The branch that is main focused on machine learning and classic vision methods for parsing images of text data. For instance, using OpenCV line detection
algorithms for detecting tables, rows, columns and so on.

### ttocr-dl
> Use virtual env `ttocrdl39`

The branch that is main focused on deep learning methods for parsing the images of text data. For instance, using
[MS Layout Parser](https://github.com/Layout-Parser/layout-parser) for extracting table or table structure.

## Remarks
1. **OCR**: Note that OCRing itself is never considered a problem to be solved and I assume, always we can OCR a "clean" image. ("clean" image means an image that does not contain any noise or other issues that OCR can't see the text.). Given this fact, OCR engines are prebuilt and considered no fine tuning is needed. E.g. I only rely on tesseract or anything similar without any level of tuning or training.

2. **Environments**: Since DL and ML libraries with classic approaches make the environment super big and might cause conflicts, it is better to use separate virtual environments for each branch.
