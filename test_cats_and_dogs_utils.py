import pytest
from cats_and_dogs_utils import *

def test_count_files():
    input_path = "cats_and_dogs/test"
    assert isinstance(count_files_in_dir(input_path),int)
    assert count_files_in_dir(input_path) > 0

def test_count_files_deep_dir():
    input_path = "cats_and_dogs/train"
    assert isinstance(count_files_in_dir(input_path),int)
    assert count_files_in_dir(input_path) > 0

