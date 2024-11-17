import re
import numpy as np
import onnx
from onnx import helper, TensorProto
import codecs
import onnxruntime as ort

# Map ONNX data type integers to NumPy data types
onnx_to_numpy_dtype = {
        1: np.float32,  # FLOAT
        2: np.uint8,    # UINT8
        3: np.int8,     # INT8
        4: np.uint16,   # UINT16
        5: np.int16,    # INT16
        6: np.int32,    # INT32
        7: np.int64,    # INT64
        8: str,         # STRING
        9: np.bool_,    # BOOL
        10: np.float16, # FLOAT16
        11: np.double,  # DOUBLE
    }

def parse_attribute(attr_name, lines):
    values = []
    attr_type = None
    tensor_data = None
    tensor_data_type = None
    tensor_raw_data = None
    tensor_dims = []
    in_tensor_block = False


    for line in lines:
        line = line.strip()

        if line.startswith("ints:"):
            # Collect all integers in a list
            values.extend([int(x) for x in line.split("ints:")[1].strip().split()])
            attr_type = "INTS"
        elif line.startswith("floats:"):
            # Collect all floats in a list
            values.extend([float(x) for x in line.split("floats:")[1].strip().split()])
            attr_type = "FLOATS"
        elif line.startswith("i:"):
            # Single integer value
            values = int(line.split("i:")[1].strip())
            attr_type = "INT"
        elif line.startswith("f:"):
            # Single float value
            values = float(line.split("f:")[1].strip())
            attr_type = "FLOAT"
        elif line.startswith("s:"):
            # String value
            values = line.split("s:")[1].strip().strip('"')
            attr_type = "STRING"
        elif line.startswith("t {"):
            # Start of a tensor block
            in_tensor_block = True
        elif in_tensor_block:
            # Parse tensor properties
            if line.startswith("dims:"):
                tensor_dims.append(int(line.split("dims:")[1].strip()))
            elif line.startswith("data_type:"):
                tensor_data_type = int(line.split("data_type:")[1].strip())
            elif line.startswith("raw_data:"):
                raw_data_str = line.split("raw_data:")[1].strip().strip('"')
                tensor_raw_data = raw_data_str.encode("latin1").decode("unicode_escape").encode("latin1")  # Convert to bytes
            elif line.startswith("}"):
                # End of tensor block
                in_tensor_block = False
                attr_type = "TENSOR"

    # Create the attribute based on its type
    if attr_type == "INTS":
        return helper.make_attribute(attr_name, values)
    elif attr_type == "FLOATS":
        return helper.make_attribute(attr_name, values)
    elif attr_type == "INT":
        return helper.make_attribute(attr_name, values)
    elif attr_type == "FLOAT":
        return helper.make_attribute(attr_name, values)
    elif attr_type == "STRING":
        return helper.make_attribute(attr_name, values)
    elif attr_type == "TENSOR":
        # Use the mapping to get NumPy data type
        np_data_type = onnx_to_numpy_dtype.get(tensor_data_type)
        if np_data_type is None:
            raise ValueError(f"Unsupported tensor data type: {tensor_data_type}")

        # Decode raw_data into a NumPy array
        tensor_data = np.frombuffer(tensor_raw_data, dtype=np_data_type)

        # Reshape if dimensions are provided, or treat as a scalar
        if tensor_dims:
            tensor_data = tensor_data.reshape(tensor_dims)
        elif tensor_data.size == 1:
            tensor_data = tensor_data.item()  # Extract scalar

        # Create TensorProto
        tensor = helper.make_tensor(
            name=attr_name,
            data_type=tensor_data_type,
            dims=tensor_dims,
            vals=tensor_data.flatten().tolist() if tensor_dims else [tensor_data],
        )
        # Create AttributeProto wrapping the TensorProto
        return helper.make_attribute(attr_name, tensor)

    else:
        raise ValueError(f"Unknown attribute type or format in lines: {lines}")

        
def log_model_structure(onnx_model_path, log_file_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    with open(log_file_path, "w") as log_file:
        # Log input information
        log_file.write("IR_VERSION: {}\n".format(model.ir_version))
        log_file.write("PRODUCER: {}\n".format(model.producer_name))
        log_file.write("PRODUCER_VERSION: {}\n".format(model.producer_version))
        log_file.write("DOMAIN: {}\n".format(model.domain))
        log_file.write("MODEL_VERSION: {}\n".format(model.model_version))
        log_file.write("OPSET_IMPORT: {}\n".format(str({opset.domain: opset.version for opset in model.opset_import})))
        for input_tensor in graph.input:
            log_file.write("Input: {}; Shape: {}; Type: {}\n".format(
                input_tensor.name, 
                str(input_tensor.type.tensor_type.shape.dim).replace('\n', ''),
                input_tensor.type.tensor_type.elem_type
            ))

        # Log output information
        for output_tensor in graph.output:
            log_file.write("Output: {}; Shape: {}; Type: {}\n".format(
                output_tensor.name,
                str(output_tensor.type.tensor_type.shape.dim).replace('\n', ''), 
                output_tensor.type.tensor_type.elem_type
            ))

        # Log initializers
        for initializer in graph.initializer:
            log_file.write("Initializer: {}; Shape: {}; Type: {}; Raw Data: {}\n".format(
                initializer.name, 
                initializer.dims,
                initializer.data_type,
                initializer.raw_data if len(initializer.raw_data)<50 else ""
            ))

        # Log each node's information
        for node in graph.node:
            log_file.write("Node: {}; Name: {}\n".format(node.op_type, node.name))
            log_file.write("  Inputs: {}\n".format(node.input))
            log_file.write("  Outputs: {}\n".format(node.output))
            log_file.write("  Attributes:\n")
            for attr in node.attribute:
                log_file.write("    - {}: {}\n".format(attr.name, attr))

def parse_log_and_reconstruct(log_file_path, output_model_path):
    nodes = []
    inputs = []
    outputs = []
    initializers = []
    
    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()
        cleaned_lines = []
        current_node = None
        line_buffer = []  # To handle multiline attributes
        for line in lines:
            if  "List inputs:" in line:
                break
            if " - DEBUG - " in line:
                line = line.split(" - DEBUG - ", 1)[1]
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        in_tensor_block = False
        for line in cleaned_lines:
            line = line.strip()
            if line.startswith("IR_VERSION:"):
                ir_version = int(line.split(": ")[1])
            elif line.startswith("PRODUCER:"):
                producer_name = line.split(": ")[1]
            elif line.startswith("PRODUCER_VERSION:"):
                producer_version = line.split(": ")[1]
            elif line.startswith("DOMAIN:"):
                domain_line = line.split(": ")
                domain = domain_line[1] if len(domain_line) > 1 else ""
            elif line.startswith("MODEL_VERSION:"):
                model_version = int(line.split(": ")[1])
            elif line.startswith("OPSET_IMPORT:"):
                pass
            
            elif line.startswith("Input:"):
                # Parse input tensor
                parts = line.split("; ")
                name = parts[0].split(": ")[1]
                shape = [int(digit) for digit in re.findall(r'dim_value: (\d+)', parts[1])]
                dtype = int(parts[2].split(": ")[1])
                inputs.append(helper.make_tensor_value_info(name, dtype, shape))
                
            elif line.startswith("Output:"):
                # Parse output tensor
                parts = line.split("; ")
                name = parts[0].split(": ")[1]
                shape = [int(digit) for digit in re.findall(r'dim_value: (\d+)', parts[1])]
                dtype = int(parts[2].split(": ")[1])
                outputs.append(helper.make_tensor_value_info(name, dtype, shape))
                
            elif line.startswith("Initializer:"):
                # Parse initializer
                parts = line.split("; ")
                name = parts[0].split(": ")[1]
                shape_str = parts[1].split(": ")[1].strip("[]")
                dtype = int(parts[2].split(": ")[1])
                raw_data_str = parts[3].split(": ")
                # Check if shape is scalar (empty string indicates shape is [])
                if shape_str:
                    shape = [int(dim) for dim in shape_str.split(",")]
                else:
                    shape = []  # Scalar case

                if len(raw_data_str) > 1:
                    raw_data_str = raw_data_str[1].strip()[2:-1]
                    raw_data = codecs.escape_decode(raw_data_str.replace('\\\\', '\\'))[0]

                    values = np.frombuffer(raw_data, dtype=onnx_to_numpy_dtype.get(dtype, np.float32)).tolist()
                else:
                    # Generate random values for the initializer
                    np_dtype = onnx_to_numpy_dtype.get(dtype, np.float32)
                    
                    if shape:  # Tensor case
                        values = np.random.rand(*shape).astype(np_dtype).flatten().tolist()
                    else:  # Scalar case
                        values = [np.random.rand()]  # Single random value
                
                initializer = helper.make_tensor(name, dtype, shape, values)
                initializers.append(initializer)
                
            elif line.startswith("Node:"):
                # Parse node information
                if current_node:
                    # If there's an existing node, add it to the nodes list
                    nodes.append(current_node)
                    
                # Start a new node
                parts = line.split("; ")
                op_type = parts[0].split(": ")[1]
                node_name = parts[1].split(": ")[1]
                current_node = {
                    "op_type": op_type,
                    "name": node_name,
                    "inputs": [],
                    "outputs": [],
                    "attributes": {}
                }
                
            elif line.startswith("Inputs:"):
                # Parse node input names
                current_node["inputs"] = line.split(": ")[1].strip('[]').replace("'", "").strip().split(",")
                current_node["inputs"] = [inp.strip() for inp in current_node["inputs"]] if current_node["inputs"] != [''] else []
                
            elif line.startswith("Outputs:"):
                # Parse node output names
                current_node["outputs"] = line.split(": ")[1].strip('[]').replace("'", "").strip().split(",")
                current_node["outputs"] = [out.strip() for out in current_node["outputs"]] if current_node["outputs"] != [''] else []
                
            elif line.startswith("-"):
                # Start accumulating attribute lines
                attr_name = line.split(":")[0].split(" ")[-1]
                line_buffer = [line]  # Start a new buffer with the first line

            elif line.startswith("type:") or line.startswith("ints:") or line.startswith("floats:") or line.startswith("i:") or line.startswith("f:") or line.startswith("s:"):
                # Append continuation of the attribute lines to the buffer
                line_buffer.append(line)
            elif line.startswith("t {"):
                # Capture the tensor block starting from `t {`
                line_buffer.append(line)  # Add the opening line
                in_tensor_block = True
            elif in_tensor_block:
                # Add all lines within the tensor block to the buffer
                line_buffer.append(line)
                if line.startswith("}"):
                    # End of tensor block
                    in_tensor_block = False

            else:
                # End of an attribute block, parse the accumulated buffer
                if line_buffer:
                    attr_name = line_buffer[0].split(":")[0].split(" ")[-1]
                    attr_data = parse_attribute(attr_name, line_buffer[1:])
                    current_node["attributes"][attr_name] = attr_data
                    line_buffer = []
                
        # Add the last node if it exists
        if current_node:
            nodes.append(current_node)

    # Create ONNX nodes
    onnx_nodes = []
    for node_info in nodes:
        attributes = node_info["attributes"]
        node = helper.make_node(
            node_info["op_type"],
            inputs=node_info["inputs"],
            outputs=node_info["outputs"],
            name=node_info["name"]
        )
        node.attribute.extend(attributes.values())
        onnx_nodes.append(node)

    # Construct the ONNX graph and model
    graph = helper.make_graph(
        onnx_nodes,
        "reconstructed_model",
        inputs,
        outputs,
        initializers
    )
    
    model = helper.make_model(graph, producer_name="log_parser")
    model.ir_version = ir_version
    model.producer_name = producer_name
    model.producer_version = producer_version
    model.domain = domain
    model.model_version = model_version
    model.opset_import.extend([helper.make_operatorsetid("", 12), helper.make_operatorsetid("aimet_torch", 1)])


    check = onnx.checker.check_model(model)
    inputs = [inp.name for inp in model.graph.input]
    # Save the reconstructed model
    onnx.save(model, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == '__main__':
    onnx_model_path = 'model.onnx'
    log_file_path = 'logs.txt'
    # log_model_structure(onnx_model_path, log_file_path)
    parse_log_and_reconstruct(log_file_path, 'reconstructed_model.onnx')