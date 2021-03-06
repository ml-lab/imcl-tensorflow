# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tests for functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.util import compat


class FileSystemTest(tf.test.TestCase):

  def setUp(self):
    file_system_library = os.path.join(tf.resource_loader.get_data_files_path(),
                                       "test_file_system.so")
    tf.load_file_system_library(file_system_library)

  def testBasic(self):
    with self.test_session() as sess:
      reader = tf.WholeFileReader("test_reader")
      queue = tf.FIFOQueue(99, [tf.string], shapes=())
      queue.enqueue_many([["test://foo"]]).run()
      queue.close().run()
      key, value = sess.run(reader.read(queue))
    self.assertEqual(key, compat.as_bytes("test://foo"))
    self.assertEqual(value, compat.as_bytes("AAAAAAAAAA"))


if __name__ == "__main__":
  tf.test.main()
