import re
import pandas as pd
import matplotlib.pyplot as plt

# Initialize an empty list to store the extracted data
data = []

# Open the log file
with open('logs.txt', 'r') as file:
    lines = file.readlines()

# Regular expressions to match the required lines
layer_info_pattern = re.compile(r'Testing layers up to index (\d+) \((.*?)\)')
loading_time_pattern = re.compile(r'Loading time: ([\d\.]+) seconds\.')

# Iterate through the lines to extract information
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Check if the line contains layer information
    layer_info_match = layer_info_pattern.search(line)
    if layer_info_match:
        layer_index = int(layer_info_match.group(1))
        layer_name = layer_info_match.group(2)
        
        # Look ahead for the loading time
        loading_time = None
        for j in range(i + 1, len(lines)):
            loading_line = lines[j].strip()
            loading_time_match = loading_time_pattern.search(loading_line)
            if loading_time_match:
                loading_time = float(loading_time_match.group(1))
                i = j  # Move the index to the line where loading time was found
                break
        else:
            print(f"Loading time not found for layer index {layer_index}")
        
        # Append the extracted information to the data list
        if loading_time is not None:
            data.append({
                'layer_index': layer_index,
                'layer_name': layer_name,
                'loading_time': loading_time
            })
    i += 1

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)
df.to_csv('loading_times.csv', index=False)
df.plot(x='layer_index', y='loading_time', kind='line', title='Layer Loading Times')
print("Done")
