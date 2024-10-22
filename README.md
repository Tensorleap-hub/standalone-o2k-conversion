# ONNX to Keras Model Converter
A Python tool for converting ONNX models to Keras (H5) format with support for channel format conversion. <br/>
This tool simplifies the process of converting models from ONNX to Keras while handling common conversion challenges like channel ordering.

## Features
- Convert ONNX models to Keras (.h5) format
- Support for channels-first to channels-last conversion
- Optional input/output data format transformation
- Command-line interface and programmatic usage
- Poetry dependency management

## Prerequisites

- Python 3.8-3.10
- Poetry (Python dependency management tool) [(poetry installation)](https://python-poetry.org/docs/#installation)



## Installation

### Extract the ZIP file:

``` bash
unzip onnx-keras-converter.zip
cd onnx-keras-converter
```

### Install dependencies using Poetry:
``` bash
poetry install
```
This will create a virtual environment and install all required dependencies.

## Usage
### Command Line Interface
The converter can be used directly from the command line:
```bash
poetry run python model_conversion.py path/to/your/model.onnx --transform-io
```
Arguments:
```
path/to/your/model.onnx: Path to your input ONNX model (required)
--transform-io: Flag to enable input/output data format transformation (optional)
```
### Python API

You can also use the converter programmatically in your Python code:

```python
from model_conversion import convert_onnx_to_keras
keras_model_path = convert_onnx_to_keras('model.onnx', transform_io=True)
```
Project Structure
```
onnx-keras-converter/
├── pyproject.toml        # Poetry project configuration
├── poetry.lock          # Poetry dependency lock file
├── model_conversion.py  # Main conversion script
└── example.py          # Usage example
```
## Common Issues and Solutions

### Channel Format Mismatch

If your model's predictions don't match the original ONNX model, try toggling the transform_io parameter
Channels-first (NCHW) vs channels-last (NHWC) format differences are a common source of issues


### Memory Issues

For large models, ensure you have sufficient RAM available
Consider using a machine with more memory for very large models

### Python Version

Ensure you're using Python 3.8-3.10 as specified in the dependencies
Python 3.11 is not currently supported due to TensorFlow compatibility