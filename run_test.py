import pytest

def run_tests():
    # Run the tests in test_image_basics.py
    pytest.main(["-v", "test_image_basics.py"])

if __name__ == "__main__":
    run_tests()

