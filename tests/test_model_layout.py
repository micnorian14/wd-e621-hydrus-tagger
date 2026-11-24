"""Regression test: channels-first (NCHW) models are refused right after loading.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import importlib
import os
import sys
import unittest
from unittest import mock

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
interrogate = importlib.import_module("wd-hydrus-tagger.interrogate")


class FakeInput:
    def __init__(self, shape):
        self.shape = shape
        self.name = "input"


class FakeSession:
    def __init__(self, shape):
        self._shape = shape

    def get_inputs(self):
        return [FakeInput(self._shape)]


class ModelLayoutTest(unittest.TestCase):
    def make(self):
        return interrogate.WaifuDiffusionInterrogator(
            "fake", "model.onnx", "selected_tags.csv", "fake", True, 4, repo_id="x")

    def load_with_shape(self, shape):
        tagger = self.make()
        with mock.patch("onnxruntime.InferenceSession", return_value=FakeSession(shape)), \
                mock.patch.object(interrogate.pd, "read_csv", return_value="tags") as read_csv:
            tagger.load(cpu=True)
        return tagger, read_csv

    def test_channels_first_is_refused(self):
        with self.assertRaises(click.ClickException) as ctx:
            self.load_with_shape(["batch_size", 3, 448, 448])
        self.assertEqual(
            ctx.exception.message,
            "This model uses a channels-first input layout, which is not supported yet. "
            "Support is TBD. Please choose a supported model.")

    def test_channels_last_still_loads(self):
        tagger, read_csv = self.load_with_shape(["batch_size", 448, 448, 3])
        read_csv.assert_called_once()
        self.assertEqual(tagger.tags, "tags")


if __name__ == "__main__":
    unittest.main()
