import pytest
from cats_and_dogs_utils import *

class TestCountFiles():
    def test_count_files(self):
        input_path = "data/test"
        assert isinstance(count_files_in_dir(input_path),int)
        assert count_files_in_dir(input_path) > 0

    def test_count_files_deep_dir(self):
        input_path = "data/train"
        assert isinstance(count_files_in_dir(input_path),int)
        assert count_files_in_dir(input_path) > 0

class TestFindMaxImageSize():
    def test_find_max_image_size(self):
        input_path = "data/train/cats"
        result = find_max_image_size(input_path)
        assert isinstance(result,tuple)
        assert isinstance(result[0],int)
        assert result[0] == 1023 or result[1] == 1023