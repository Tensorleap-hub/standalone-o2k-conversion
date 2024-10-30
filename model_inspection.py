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
        logging.error(f"Loading exceeded {t} seconds.")
        return False, t  # Indicate that loading time exceeded t seconds
    else:
        load_time = queue.get()
        if load_time is None:
            logging.error(f"Loading failed due to an error.")
            return False, None
        else:
            return True, load_time  # Return success status and actual loading time

def find_first_problematic_layer(model, t):
    logging.info(f"Total number of layers: {len(model.layers)}")
    logging.info("Testing each layer incrementally...")

    model_path = 'temp_model.h5'  # Temporary file to save the model
    loading_times = []
    problematic_layers = []
    initial_layer = 900
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
        success, load_time = test_loading_time(model_path, t)

        # Delete the temporary model file
        # if os.path.exists(model_path):
        #     os.remove(model_path)

        if success:
            logging.info(f"Loading succeeded in {load_time:.2f} seconds.")
            low = mid + 1
        else:
            if load_time == t:
                logging.error(f"Loading failed (exceeded maximal time of {t} seconds).")
            else:
                logging.error(f"Loading failed due to an error.")
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
        return problematic_index

    else:
        logging.info("No problematic layer found within the given time threshold.")

def search_graph_backward(model, layer_idx, t):
    logging.info("Starting backward graph traversal from the problematic layer.")

    # Keep track of visited layers to avoid cycles
    visited_layers = set()
    # Stack for DFS traversal
    stack = []
    problematic_layers = set()

    # Start from the problematic layer
    target_layer = model.layers[layer_idx]
    stack.append(target_layer)

    while stack:
        current_layer = stack.pop()
        layer_name = current_layer.name

        if layer_name in visited_layers:
            continue

        visited_layers.add(layer_name)
        logging.info(f"Analyzing layer: {layer_name} (Type: {current_layer.__class__.__name__})")

        inbound_nodes = current_layer.inbound_nodes
        if not inbound_nodes:
            logging.info(f"Layer {layer_name} has no inbound nodes.")
            continue

        # Assuming the first inbound node (might need to iterate over all nodes if necessary)
        inbound_layers = inbound_nodes[0].inbound_layers
        inbound_layers = [inbound_layers] if not isinstance(inbound_layers, list) else inbound_layers

        while inbound_layers:
            inbound_layer = inbound_layers.pop()
            parent_output = inbound_layer.output
            parent_layer_name = inbound_layer.name
            submodel = Model(inputs=model.inputs, outputs=parent_output)

            # Save the submodel
            model_path = f"submodel_{parent_layer_name.replace('/', '$')}.h5"
            save_model(submodel, model_path)

            # Test loading time
            success, load_time = test_loading_time(model_path, t)

            # Delete the temporary model file
            if os.path.exists(model_path):
                os.remove(model_path)

            if success:
                logging.info(f"Submodel up to layer {parent_layer_name} loaded successfully in {load_time:.2f} seconds.")
                if not inbound_layers:
                    logging.info(f"Problematic layer found: {parent_layer_name}")
                    problematic_layers.add(inbound_layer.outbound_nodes[0].outbound_layer.name)
                else:
                    stack.clear()
                    stack.extend(inbound_layers)
            else:
                if load_time == t:
                    logging.error(f"Submodel up to layer {parent_layer_name} failed to load (exceeded maximal time of {t} seconds).")
                else:
                    logging.error(f"Submodel up to layer {parent_layer_name} failed to load due to an error.")
                stack.append(parent_output._keras_history.layer)  # Add the parent layer to the stack
                break

def main(onnx_model_path, t):
    keras_model = get_keras_model(onnx_model_path)
    first_layer_idx = find_first_problematic_layer(keras_model, t)
    search_graph_backward(keras_model, first_layer_idx, t)

if __name__ == '__main__':
    t = 25  # Time threshold in seconds
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
