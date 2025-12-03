import os
import re
import json
import gzip
from typing import Iterable, Dict

# Read JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

# Read JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Write JSONL file
def write_jsonl(
    filename: str, data: Iterable[Dict], append: bool = False, drop_builtin: bool = True
):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if drop_builtin:
                        x = {k: v for k, v in x.items() if not k.startswith("_")}
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if drop_builtin:
                    x = {k: v for k, v in x.items() if not k.startswith("_")}
                fp.write((json.dumps(x) + "\n").encode("utf-8"))