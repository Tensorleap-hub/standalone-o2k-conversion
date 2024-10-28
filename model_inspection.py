import logging
import logging.handlers
import multiprocessing
import os
import sys
import time
import onnx
from onnx2kerastl import onnx_to_keras
from onnx2kerastl.customonnxlayer import onnx_custom_objects_map
from keras_data_format_converter import convert_channels_first_to_last
from keras.models import Model
import tensorflow as tf

# Global log queue
log_queue = multiprocessing.Queue()

def configure_main_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs to 'logs.txt'
    fh = logging.FileHandler('logs.txt', mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    # Create a queue listener to handle logs from subprocesses
    queue_listener = logging.handlers.QueueListener(log_queue, fh)
    queue_listener.start()

    return queue_listener

def configure_subprocess_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a handler that sends log messages to the queue
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

def redirect_stdout_stderr():
    class StreamToLogger(object):
        """
        Fake file-like stream object that redirects writes to a logger instance.
        """
        def __init__(self, logger, log_level=logging.INFO):
            self.logger = logger
            self.log_level = log_level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())

        def flush(self):
            pass

    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

def get_keras_model(onnx_model_path):
    logging.info(f"Loading ONNX model from {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    input_features = [inp.name for inp in onnx_model.graph.input]
    logging.info(f"Converting ONNX model to Keras model")
    keras_model = onnx_to_keras(onnx_model, input_features).converted_model
    keras_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)
    return keras_model

def trim_model(model, index):
    """
    Trims the Keras model up to the layer at the specified index.
    """
    # Get the output of the last layer to include
    layer_output = model.layers[index].output
    # Create a new model from the original inputs to the specified layer's output
    trimmed_model = Model(inputs=model.inputs, outputs=layer_output)
    return trimmed_model

def save_model(model, model_path):
    model.save(model_path)

def load_model(queue, model_path):
    configure_subprocess_logger()
    redirect_stdout_stderr()
    start_time = time.time()
    try:
        model = tf.keras.models.load_model(model_path, 
                                           custom_objects=onnx_custom_objects_map,
                                           compile=False)
        load_time = time.time() - start_time
        logging.info(f"Model loaded in {load_time:.2f} seconds.")
        queue.put(load_time)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        queue.put(None)

def test_loading_time(model_path, t):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=load_model, args=(queue, model_path))
    p.start()
    p.join(timeout=t)
    if p.is_alive():
        p.terminate()
        p.join()
        logging.info(f"Loading exceeded {t} seconds.")
        return False  # Loading time exceeded t seconds
    else:
        load_time = queue.get()
        if load_time is None:
            logging.error(f"Loading failed due to an error.")
            return False
        else:
            return True  # Loading completed successfully within t seconds

def find_problematic_layer(model, t):
    logging.info(f"Total number of layers: {len(model.layers)}")
    logging.info("Looking for problematic layer...")

    low = 0
    high = len(model.layers) - 1
    model_path = 'temp_model.h5'  # Temporary file to save the model

    while low <= high:
        mid = (low + high) // 2
        logging.info(f"Testing layers up to index {mid} ({model.layers[mid].name})")

        # Trim the model up to the mid index
        trimmed_model = trim_model(model, mid)

        # Save the trimmed model
        save_model(trimmed_model, model_path)

        # Test loading time
        success = test_loading_time(model_path, t)

        # Delete the temporary model file
        if os.path.exists(model_path):
            os.remove(model_path)

        if success:
            logging.info(f"Loading succeeded within {t} seconds.")
            low = mid + 1
        else:
            high = mid - 1

    problematic_index = low
    if problematic_index < len(model.layers):
        logging.info(f"Problematic layer suspected at index {problematic_index} ({model.layers[problematic_index].name})")
        logging.info(f"Layer at index {problematic_index - 2}: {model.layers[problematic_index - 2].name}")
        logging.info(f"Layer at index {problematic_index - 1}: {model.layers[problematic_index - 1].name}")
        logging.info(f"Layer at index {problematic_index + 1}: {model.layers[problematic_index + 1].name}")
        logging.info(f"Layer at index {problematic_index + 2}: {model.layers[problematic_index + 2].name}")

        logging.info(f"Problematic layer input: {model.layers[problematic_index].input}")
        logging.info(f"Problematic layer output: {model.layers[problematic_index].output}")
    else:
        logging.info("No problematic layer found within the given time threshold.")

def main(onnx_model_path, t):
    keras_model = get_keras_model(onnx_model_path)
    find_problematic_layer(keras_model, t)

if __name__ == '__main__':
    t = 50  # Time threshold in seconds
    onnx_model_path = 'mod_efficiency.onnx'

    # Configure logging and start the queue listener
    queue_listener = configure_main_logger()

    # Redirect stdout and stderr to logging
    redirect_stdout_stderr()

    try:
        main(onnx_model_path, t)
    finally:
        # Stop the queue listener and clean up
        queue_listener.stop()
