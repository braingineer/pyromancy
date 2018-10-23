# coding=utf-8
import itertools
import os
from glob import glob


def get_filenames(root, ext="json"):
    ext = "*.{}".format(ext)
    return glob(os.path.join(root, ext))


def file_path_exists(file_path, file_system="local"):
    if file_system == "local":
        return os.path.exists(file_path)
    else:
        raise Exception("unknown file system: {}".format(file_system))


def expand_options(options):
    out = []
    for key, value_list in options.items():
        out.append([])
        for value in value_list:
            out[-1].append((key, value))

    return list(itertools.product(*out))
