# -*- coding: utf-8 -*-
import os
import shutil

def convert_type(x, type):
    if type == 'int':
        return int(x)
    if type == 'long':
        return long(x)
    if type == 'float':
        return float(x)
    return x


def list_files(filepath, suffix=None, isdepth=True):
    files = []
    for fpathe, dirs, fs in os.walk(filepath):
        for f in fs:
            if suffix is None or f.endswith(suffix):
                files.append(os.path.join(fpathe, f))
        if isdepth == False:
            break
    return files

def get_filename_from_absolute_path(filepath, retain_suffix=True):
    res = None
    if filepath.find('/') >= 0:
        items = filepath.split('/')
        res = items[-1]
    elif filepath.find('\\') >= 0:
        items = filepath.split('/')
        res = items[-1]
    if res is not None:
        if retain_suffix == False:
            idx = res.find('.')
            if idx >= 0:
                res = res[0:idx]
    return res


def load_map(path, key_type, value_type, split_tag='\t'):
    res = {}
    for line in open(path, 'r'):
        items = line.strip().split(split_tag)
        key = convert_type(items[0], key_type)
        value = convert_type(items[1], value_type)
        res[key] = value

    return res

def load_reverse_map(path, key_type, value_type, split_tag='\t'):
    res = {}
    for line in open(path, 'r'):
        items = line.strip().split(split_tag)
        value = convert_type(items[0], value_type)
        key = convert_type(items[1], key_type)
        res[key] = value

    return res

def load_list(path):
    with open(path, 'r') as reader:
        lines = reader.readlines()
    res = [s.strip() for s in lines]
    return res


def dump_map(path, map, split_tag='\t'):
    with open(path, 'w') as writer:
        for key, value in map.items():
            line = str(key) + split_tag + str(value) + '\n'
            writer.write(line)


def detect_and_create_dir(dir):
    if os.path.exists(dir) == False:
        os.makedirs(dir)

def detect_and_delete_empty_dir(dir):
    if os.path.exists(dir) == True:
        os.removedirs(dir)

def detect_and_delete_dir(dir):
    if os.path.exists(dir) == True:
        shutil.rmtree(dir)
