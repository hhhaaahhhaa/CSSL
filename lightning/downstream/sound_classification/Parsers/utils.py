import os

from .parser import DataParser


def read_queries_from_txt(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == '\n':
                continue
            n, c = line.strip("\n").split("|")
            res.append({
                "basename": n,
                "label": c,
            })
    return res


def write_queries_to_txt(data_parser: DataParser, queries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_parser.label.read_all()
    lines = []
    for query in queries:
        try:
            line = [query["basename"], data_parser.label.read_from_query(query)["class"]]
            lines.append(line)
        except:
            print("Failed: ", query)
            raise
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("|".join(line))
            f.write('\n')
