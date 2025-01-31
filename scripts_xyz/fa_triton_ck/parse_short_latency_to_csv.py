# read a short file, get its config, and write its output in a csv format

import csv
import argparse

def parse_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    configs = []
    latencys = []
    current_config = None

    for line in lines:
        if line.startswith("=="):
            if current_config is not None:
                # If there was a previous config without latency, mark it as error
                configs.append(current_config)
                latencys.append(["error"])
#===================== RUNNING /data/Meta-Llama-3-70B-Instruct 128 128 1 tp=8 ===================================================
# Avg latency: 1.6046842340030707 seconds


            current_config = line.strip().split(" ")[2:7]  # Split and remove the "=="
            # import pdb;pdb.set_trace()
        elif line.startswith("Avg latency"):
            if current_config is not None:
                latency_values = line.split(" ")
                latency_numbers=[ latency_values[2] ]
                # .strip().split()
                # value 1 3 5
                latencys.append(latency_numbers)
                configs.append(current_config)
                current_config = None

    # If the last config has no latency, mark it as error
    if current_config is not None:
        configs.append(current_config)
        latencys.append(["error"])

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'input-len', 'output-len', "batch-size", "TP", 'latency (s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for config, latency in zip(configs, latencys):
            row = {
                fieldnames[0]: config[0],
                fieldnames[1]: config[1],
                fieldnames[2]: config[2],
                fieldnames[3]: config[3],
                fieldnames[4]: config[4],
                fieldnames[5]: latency[0],
            }
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a file and extract configs and latency values.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    parse_file(args.input_file, args.output_file)

