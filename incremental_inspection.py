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
        logging.warning(f"Loading exceeded {t} seconds.")
        return t  # Loading time exceeded t seconds
    else:
        load_time = queue.get()
        if load_time is None:
            logging.error(f"Loading failed due to an error.")
            return None
        else:
            return load_time  # Actual loading time

def find_problematic_layers(model, t):
    logging.info(f"Total number of layers: {len(model.layers)}")
    logging.info("Testing each layer incrementally...")

    model_path = 'temp_model.h5'  # Temporary file to save the model
    loading_times = []
    problematic_layers = []
    initial_layer = 0
    for idx in range(len(model.layers) - initial_layer):
        idx = idx + initial_layer - 1
        layer_name = model.layers[idx].name
        logging.info(f"Testing layers up to index {idx} ({layer_name})")

        try:
            # Trim the model up to the idx index
            trimmed_model = trim_model(model, idx)

            # Save the trimmed model
            save_model(trimmed_model, model_path)

            # Test loading time
            load_time = test_loading_time(model_path, t)

            # Delete the temporary model file
            if os.path.exists(model_path):
                os.remove(model_path)

            # Record the loading time
            if load_time is not None:
                logging.info(f"Loading time: {load_time:.2f} seconds.")
                loading_times.append((idx, layer_name, load_time))
                if load_time >= t:
                    # Record the layer as problematic
                    logging.warning(f"Layer {layer_name} causes loading time >= {t} seconds.")
                    problematic_layers.append((idx, layer_name, load_time))
            else:
                logging.info(f"Loading failed for layer index {idx} ({layer_name}).")
                loading_times.append((idx, layer_name, None))
                problematic_layers.append((idx, layer_name, None))

        except Exception as e:
            logging.error(f"Error processing layer {layer_name}: {e}")
            loading_times.append((idx, layer_name, None))
            problematic_layers.append((idx, layer_name, None))
            if os.path.exists(model_path):
                os.remove(model_path)

        # Clean up
        tf.keras.backend.clear_session()

    # Log problematic layers
    if problematic_layers:
        logging.info("Problematic layers:")
        for idx, layer_name, load_time in problematic_layers:
            logging.info(f"Layer index: {idx}, Name: {layer_name}, Loading time: {load_time if load_time is not None else 'Failed'}")
            logging.info(f"Layer input: {model.layers[idx].input}")
            logging.info(f"Layer output: {model.layers[idx].output}")
    else:
        logging.info("No problematic layers found within the given time threshold.")

    # Optionally, save all loading times to a file or log
    logging.info("All layer loading times:")
    for idx, layer_name, load_time in loading_times:
        logging.info(f"Layer index: {idx}, Name: {layer_name}, Loading time: {load_time if load_time is not None else 'Failed'}")

def main(onnx_model_path, t):
    keras_model = get_keras_model(onnx_model_path)
    find_problematic_layers(keras_model, t)

if __name__ == '__main__':
    t = 120  # Time threshold in seconds
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
