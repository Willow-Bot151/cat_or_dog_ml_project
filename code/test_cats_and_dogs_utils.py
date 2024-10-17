import pytest
from cats_and_dogs_utils import *
import os

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
    def test_find_max_image_size_1(self):
        input_path = "data/train/cats"
        result = find_max_image_size(input_path)
        assert isinstance(result,tuple)
        assert isinstance(result[0],int)
        assert result[0] == 1023 or result[1] == 1023
    def test_find_max_image_size_2(self):
        input_path = "data/train"
        result = find_max_image_size(input_path)
        assert isinstance(result,tuple)
        assert isinstance(result[0],int)
        assert result[0] == 1023 or result[1] == 1023
    def test_find_max_image_size_3(self):
        input_path = "data/test"
        result = find_max_image_size(input_path)
        assert isinstance(result,tuple)
        assert isinstance(result[0],int)
        assert result[0] == 1023 or result[1] == 500

class TestCleanTempImagePath():
    def test_clean_temp_image_path_func_works_on_empty_dir(self):
        input_test_path = "testpath"
        clean_temp_image_path(input_test_path)
        assert os.path.exists(os.path.join("temp",input_test_path))
    def test_func_deletes_deep_dirs(self):
        input_test_path = "testpath"
        with open("temp/testpath/testfile.txt", "w") as f:
            f.write("testing")
        clean_temp_image_path(input_test_path)
        assert not os.path.exists("temp/testpath/testfile")

# class TestShrinkAndSaveImageFunction():
#     def test_function_works(self):
#         os.mkdir


## test create ds by mocking directory
