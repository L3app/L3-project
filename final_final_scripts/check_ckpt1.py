#!/usr/bin/env python

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name='tmp/model.ckpt', tensor_name='', all_tensors=False)
