#Match the input and output images and patch them side by side. The paired images should have a similar format as the Facade data used in Isola et al. (Left is input, and right is output.)
#Could be optimized by multiproccesing etc.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess
import numpy as np
import threading
import time

edge_pool = None

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True,
                    help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
parser.add_argument("--operation", required=True,
                    choices=["grayscale", "resize", "blank", "combine",
                             "edges"])
parser.add_argument("--workers", type=int, default=1, help="number of workers")
# resize
parser.add_argument("--pad", action="store_true",
                    help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256,
                    help="size to use for resize operation")
# combine
parser.add_argument("--b_dir", type=str,
                    help="path to folder containing B images for combine operation")
a = parser.parse_args()


def combine(src, src_path):
    if a.b_dir is None:
        raise Exception("missing b_dir")

    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".npz"]:
        sibling_path = os.path.join(a.b_dir, basename + ext)
        if os.path.exists(sibling_path):
            siblingfile = np.load(sibling_path)
            sibling=siblingfile['imagevar']
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")
    
    return np.concatenate([src, sibling], axis=1)

net = None


def run_caffe(src):
    # lazy load caffe and create net
    global net
    if net is None:
        # don't require caffe unless we are doing edge detection
        os.environ["GLOG_minloglevel"] = "2"  # disable logging from caffe
        import caffe
        # using this requires using the docker image or assembling a bunch of dependencies
        # and then changing these hardcoded paths
        net = caffe.Net("/opt/caffe/examples/hed/deploy.prototxt",
                        "/opt/caffe/hed_pretrained_bsds.caffemodel", caffe.TEST)

    net.blobs["data"].reshape(1, *src.shape)
    net.blobs["data"].data[...] = src
    net.forward()
    return net.blobs["sigmoid-fuse"].data[0][0, :, :]



def process(src_path, dst_path):
    #changed from im.load:
    srcfile = np.load(src_path)
    src=srcfile['imagevar']

    if a.operation == "combine":
        dst = combine(src, src_path)
    else: #compared to the original process_TF2.py, deleted all the other options here. 
        raise Exception("invalid operation")

    #im.save(dst, dst_path)
    np.save(dst_path,dst)


complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0


def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" %
              (num_complete, total, rate, elapsed // 60, elapsed % 60,
               remaining // 60, remaining % 60))

        last_complete = now


def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    src_paths = []
    dst_paths = []

    skipped = 0
    for src_path in os.listdir(a.input_dir):
        src_path=os.path.join(a.input_dir,'',src_path)#ad-hoc
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(a.output_dir, name + ".npy")
        
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)

    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)

    print("processing %d files" % total)

    global start
    start = time.time()
   
    if a.workers == 1:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path)
                complete()


main()
