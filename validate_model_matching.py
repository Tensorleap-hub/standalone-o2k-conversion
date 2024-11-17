import onnx
import numpy as np

def compare_models(original_model_path, reconstructed_model_path):
    def extract_graph_info(model):
        graph_info = {
            "inputs": [],
            "outputs": [],
            "initializers": {},
            "nodes": []
        }
        graph = model.graph
        
        # Extract inputs
        for input_tensor in graph.input:
            graph_info["inputs"].append({
                "name": input_tensor.name,
                "shape": [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim],
                "type": input_tensor.type.tensor_type.elem_type
            })
        
        # Extract outputs
        for output_tensor in graph.output:
            graph_info["outputs"].append({
                "name": output_tensor.name,
                "shape": [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim],
                "type": output_tensor.type.tensor_type.elem_type
            })
        
        # Extract initializers (exclude raw data/values)
        for initializer in graph.initializer:
            graph_info["initializers"][initializer.name] = {
                "shape": list(initializer.dims),
                "type": initializer.data_type
            }
        
        # Extract nodes (exclude raw data/values in attributes)
        for node in graph.node:
            filtered_attributes = {}
            for attr in node.attribute:
                if attr.name == "value":
                    # For tensor attributes, include only shape and type
                    tensor = onnx.helper.get_attribute_value(attr)
                    filtered_attributes[attr.name] = {
                        "shape": list(tensor.dims),
                        "type": tensor.data_type
                    }
                else:
                    # Include other attributes as-is
                    filtered_attributes[attr.name] = onnx.helper.get_attribute_value(attr)
            
            graph_info["nodes"].append({
                "op_type": node.op_type,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": filtered_attributes
            })
        
        return graph_info

    # Load both models
    original_model = onnx.load(original_model_path)
    reconstructed_model = onnx.load(reconstructed_model_path)
    
    # Extract graph information
    original_info = extract_graph_info(original_model)
    reconstructed_info = extract_graph_info(reconstructed_model)
    
    # Compare inputs
    if original_info["inputs"] != reconstructed_info["inputs"]:
        print("Mismatch in inputs!")
        return False
    
    # Compare outputs
    if original_info["outputs"] != reconstructed_info["outputs"]:
        print("Mismatch in outputs!")
        return False
    
    # Compare initializers (ignoring values)
    if original_info["initializers"] != reconstructed_info["initializers"]:
        print("Mismatch in initializers (shapes or types)!")
        return False
    
    # Compare nodes
    if len(original_info["nodes"]) != len(reconstructed_info["nodes"]):
        print("Mismatch in number of nodes!")
        return False
    
    for orig_node, recon_node in zip(original_info["nodes"], reconstructed_info["nodes"]):
        if orig_node != recon_node:
            print(f"Mismatch in node: {orig_node['name']}")
    
    print("The models are identical in structure!")
    return True

if __name__ == "__main__":

    original_model_path = "model.onnx"
    reconstructed_model_path = "reconstructed_model.onnx"
    is_identical = compare_models(original_model_path, reconstructed_model_path)
    print("Verification Result:", is_identical)
