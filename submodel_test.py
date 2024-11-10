from keras.models import Model
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last


def get_keras_model(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    input_features = [inp.name for inp in onnx_model.graph.input]
    keras_model = onnx_to_keras(onnx_model, input_features).converted_model
    keras_model = convert_channels_first_to_last(keras_model, should_transform_inputs_and_outputs=False, verbose=True)
    return keras_model

model = get_keras_model('mod_efficiency.onnx') 
target_layer_idx = 1131

target_layer = model.layers[target_layer_idx]

layer_inputs = [layer_input for layer_input in target_layer._inbound_nodes[0].call_args]

# Create the new model
new_model = Model(inputs=model.inputs, outputs=model.layers[target_layer_idx].output)

new_model.save('new_model.h5')

new_model.summary()
