# read a short file, get its config, and write its output in a csv format

import csv
import argparse

def parse_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    configs = []
    throughputs = []
    current_config = None

    for line in lines:
        if line.startswith("=="):
            if current_config is not None:
                # If there was a previous config without throughput, mark it as error
                configs.append(current_config)
                throughputs.append(["error", "error", "error"])
#===================== RUNNING /data/Meta-Llama-3-70B-Instruct 128 2048 tp=4 ===================================================
#Throughput: 1.94 requests/s, 4224.40 total tokens/s, 3975.90 output tokens/s

            current_config = line.strip().split(" ")[3:6]  # Split and remove the "=="
            # import pdb;pdb.set_trace()
        elif line.startswith("Throughput"):
            if current_config is not None:
                throughput_values = line.split(" ")
                throughput_numbers=[ throughput_values[1], throughput_values[3], throughput_values[6] ]
                # .strip().split()
                # value 1 3 5
                throughputs.append(throughput_numbers)
                configs.append(current_config)
                current_config = None

    # If the last config has no throughput, mark it as error
    if current_config is not None:
        configs.append(current_config)
        throughputs.append(["error", "error", "error"])

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Input-len', 'Output-len', "TP", 'Req/s', 'Total_tokens/s', 'Output_tokens/s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for config, throughput in zip(configs, throughputs):
            row = {
                'Input-len': config[0],
                'Output-len': config[1],
                'TP': config[2],
                'Req/s': throughput[0],
                'Total_tokens/s': throughput[1],
                'Output_tokens/s': throughput[2]
            }
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a file and extract configs and throughput values.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    parse_file(args.input_file, args.output_file)

