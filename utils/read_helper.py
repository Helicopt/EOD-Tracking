#!/usr/bin/env python
# -*- coding:utf8 -*-
# Standard Library
import sys

# Import from third library
import cv2
from os.path import realpath
# cv2.ocl.setUseOpenCL(False)
import numpy as np
import json
import os

try:
    from petrel_client.client import Client
except ImportError:
    Client = None

pyv = sys.version[0]


ceph_conf_path = '~/.s3cfg'


def bytes_to_img(value):
    img = None

    img_array = np.frombuffer(value, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    assert img is not None
    return img


class ImageHelper(object):
    def __init__(self):
        assert Client is not None, 'petrel_client module not found. please install before using ceph'
        self.client = Client(conf_path=ceph_conf_path)

    def imread(self, path):
        try:
            value = self.client.get(path)
            img = bytes_to_img(value)
        except:  # noqa
            value = self.client.get(path, update_cache=True)
            img = bytes_to_img(value)
        return img

    def imwrite(self, path, img):
        raise ImportError


global_helper = ImageHelper()


def _imread(path):
    # logger.info(path)
    return global_helper.imread(path)


def _imwrite(path, img):
    return global_helper.imwrite(path, img)


def read_link(root):
    if 's3://' in root:
        return root

    root = realpath(root)

    if 's3:/' in root:
        root = root[1:].replace('s3:/', 's3://')
    # print(root)
    return root


def read_lines(path):
    if 's3://' in path:
        c = global_helper.client
        response = c.get(path, enable_stream=True, no_cache=True)
        for i, line in enumerate(response.iter_lines()):
            cur_line = line.decode('utf-8')
            yield json.loads(cur_line)
    else:
        for item in open(path).readlines():
            yield json.loads(item)


def read_json(path):
    # print(path)
    if 's3://' in path:
        c = global_helper.client
        strings = c.Get(path)
        return json.loads(strings)
    else:
        return json.load(open(path))
