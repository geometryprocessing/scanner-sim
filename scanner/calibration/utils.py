import os
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpinize(data):
    return {k: (np.array(v) if (type(v) is list or type(v) is tuple) else
               (numpinize(v) if type(v) is dict else v)) for k, v in data.items()}


def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    data = {"l": [1, 2, 3],
            "t": (4, 5, 6),
            "i": 7,
            "f": 8.0,
            "a": np.array([9, 10, 11]),
            "d": {"a2": np.array([12, 13, 14])}}

    print(data)

    with open("test.json", "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

    with open("test.json", "r") as f:
        data = numpinize(json.load(f))

    print(data)
