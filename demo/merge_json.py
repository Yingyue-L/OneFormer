# merge different chunk json

import json
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge json files')
    parser.add_argument('--num-chunks', help='Number of chunks', type=int)
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--task', help='Task name', default="panoptic")
    args = parser.parse_args()
    merge_json = {}
    for i in range(args.num_chunks):
        with open(os.path.join(args.output, '{}_{}_{}.json'.format(args.task, args.num_chunks, i)), 'r') as f:
            data = json.load(f)
        # merge data
        merge_json.update(data)
    with open(os.path.join(args.output, '{}.json'.format(args.task)), 'w') as f:
        json.dump(merge_json, f)
