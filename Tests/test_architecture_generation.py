import unittest
from unittest.mock import patch
from pynattas.functions.architecture_builder import generate_random_architecture_code

class TestArchitectureGeneration(unittest.TestCase):

    @patch('your_module.random.randint')
    @patch('your_module.generate_layer_code')
    @patch('your_module.generate_pooling_layer_code')
    @patch('your_module.generate_head_code')
    def test_generate_random_architecture_code(self, mock_head, mock_pool, mock_layer, mock_randint):
        # Setup mock return values
        mock_randint.return_value = 3
        mock_layer.return_value = 'L'
        mock_pool.return_value = 'P'
        mock_head.return_value = 'HeadCode'

        architecture_code = generate_random_architecture_code(5, 'test_config.ini')
        expected_code = 'LEPELEPEHeadCodeEE'
        
        self.assertEqual(architecture_code, expected_code)

if __name__ == '__main__':
    unittest.main()