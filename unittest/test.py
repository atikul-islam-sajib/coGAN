import os
import sys
import unittest


sys.path.append("./src")

from src.generator import CoupledGenerators

class UnitTest(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def test_coupleGenerator(self):
        pass
    
    
if __name__ == '__main__':
    unittest.main()