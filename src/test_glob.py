from pathlib import Path

TEST_REPO = Path("tests/test_repos/textual")

import time

def test_glob_timing():
    # Test glob("*")
    start_time = time.time()
    list(TEST_REPO.glob("*"))
    all_files_time = time.time() - start_time

    # Test glob("*.py") 
    start_time = time.time()
    list(TEST_REPO.glob("*.py"))
    py_files_time = time.time() - start_time

    print(f"Time for glob('*'): {all_files_time:.4f} seconds")
    print(f"Time for glob('*.py'): {py_files_time:.4f} seconds")

if __name__ == "__main__":
    test_glob_timing()
