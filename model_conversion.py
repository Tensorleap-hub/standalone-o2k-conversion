import argparse
import onnx
from onnx2kerastl import onnx_to_keras
from keras_data_format_converter import convert_channels_first_to_last

def convert_onnx_to_keras(onnx_model_path: str, transform_io: bool) -> str:
    """
    Convert an ONNX model to Keras format and save it as H5.
    
    Args:
        onnx_model_path (str): Path to the input ONNX model
        transform_io (bool): Whether to transform input and output data format
        
    Returns:
        str: Path where the converted model was saved
    """
    # Load ONNX model
    print("Loading ONNX model...")
    save_model_path = onnx_model_path.replace('.onnx', '.h5')
    onnx_model = onnx.load(onnx_model_path)
    
    # Extract input feature names from the model
    input_features = [inp.name for inp in onnx_model.graph.input]
    
    # Convert ONNX model to Keras
    print("Converting ONNX model to Keras...")
    keras_model = onnx_to_keras(
        onnx_model, 
        input_names=input_features,
        name_policy='attach_weights_name', 
        allow_partial_compilation=False
    ).converted_model

    # Convert from channels-first to channels-last format
    print("Converting channels from first to last...")
    final_model = convert_channels_first_to_last(
        keras_model, 
        should_transform_inputs_and_outputs=transform_io,
        verbose=True
    )

    # Save the final Keras model
    print("Saving the model...")
    final_model.save(save_model_path)
    print(f"Model saved to {save_model_path}")
    return save_model_path

def main():
    """Command line interface for the converter."""
    parser = argparse.ArgumentParser(description='Convert ONNX model to Keras')
    parser.add_argument('onnx_path', type=str, help='Path to the input ONNX model')    
    parser.add_argument(
        '--transform-io', 
        action='store_true',
        help='Whether to transform input and output data format'
    )
    args = parser.parse_args()
    
    convert_onnx_to_keras(args.onnx_path, args.transform_io)

if __name__ == '__main__':
    main()