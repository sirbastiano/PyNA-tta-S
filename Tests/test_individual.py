import pytest
import logging
from unittest.mock import patch
from pynattas.builder.individual import Individual

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@patch('pynattas.builder.individual.random.randint')
@patch('pynattas.builder.individual.layerCoder.get_task_from_config')
@patch('pynattas.builder.individual.generate_layer_code')
@patch('pynattas.builder.individual.generate_pooling_layer_code')
@patch('pynattas.builder.individual.generate_head_code')
def test_generate_random_architecture_code(mock_head, mock_pool, mock_layer, mock_task, mock_randint):
    # Setup mock return values
    mock_task.return_value = 'D'
    mock_randint.return_value = 3
    mock_layer.return_value = 'L'
    mock_pool.return_value = 'P'
    mock_head.return_value = 'HeadCode'

    logger.debug("Mock return values set up")

    individual = Individual(max_layers=5)
    architecture_code = individual.generate_random_architecture_code(5, 'test_config.ini')
    expected_code = 'LEPELEPEHeadCodeEE'
    
    logger.debug(f"Generated architecture code: {architecture_code}")
    logger.debug(f"Expected architecture code: {expected_code}")
    
    assert architecture_code == expected_code

@patch('pynattas.builder.individual.random.randint')
@patch('pynattas.builder.individual.layerCoder.get_task_from_config')
@patch('pynattas.builder.individual.generate_layer_code')
@patch('pynattas.builder.individual.generate_pooling_layer_code')
@patch('pynattas.builder.individual.generate_head_code')
def test_generate_random_architecture_code_min_layers(mock_head, mock_pool, mock_layer, mock_task, mock_randint):
    # Setup mock return values
    mock_task.return_value = 'D'
    mock_randint.return_value = 3
    mock_layer.return_value = 'L'
    mock_pool.return_value = 'P'
    mock_head.return_value = 'HeadCode'

    logger.debug("Mock return values set up")

    individual = Individual(max_layers=3)
    architecture_code = individual.generate_random_architecture_code(3, 'test_config.ini')
    expected_code = 'LEPELEPEHeadCodeEE'
    
    logger.debug(f"Generated architecture code: {architecture_code}")
    logger.debug(f"Expected architecture code: {expected_code}")
    
    assert architecture_code == expected_code

def test_architecture2chromosome():
    individual = Individual(max_layers=5)
    architecture_code = 'LEPELEPEHeadCodeEE'
    expected_chromosome = ['L', 'P', 'L', 'P', 'L', 'P', 'HeadCode']
    
    logger.debug(f"Architecture code: {architecture_code}")
    logger.debug(f"Expected chromosome: {expected_chromosome}")
    
    chromosome = individual.architecture2chromosome(architecture_code)
    
    logger.debug(f"Generated chromosome: {chromosome}")
    
    assert chromosome == expected_chromosome

def test_architecture2chromosome_empty():
    individual = Individual(max_layers=5)
    architecture_code = ''
    expected_chromosome = []
    
    logger.debug(f"Architecture code: {architecture_code}")
    logger.debug(f"Expected chromosome: {expected_chromosome}")
    
    chromosome = individual.architecture2chromosome(architecture_code)
    
    logger.debug(f"Generated chromosome: {chromosome}")
    
    assert chromosome == expected_chromosome

def test_chromosome2architecture():
    individual = Individual(max_layers=5)
    chromosome = ['L', 'P', 'L', 'P', 'L', 'P', 'HeadCode']
    expected_architecture_code = 'LPLPLPHeadCodeEE'
    
    logger.debug(f"Chromosome: {chromosome}")
    logger.debug(f"Expected architecture code: {expected_architecture_code}")
    
    architecture_code = individual.chromosome2architecture(chromosome)
    
    logger.debug(f"Generated architecture code: {architecture_code}")
    
    assert architecture_code == expected_architecture_code

def test_chromosome2architecture_empty():
    individual = Individual(max_layers=5)
    chromosome = []
    expected_architecture_code = 'EE'
    
    logger.debug(f"Chromosome: {chromosome}")
    logger.debug(f"Expected architecture code: {expected_architecture_code}")
    
    architecture_code = individual.chromosome2architecture(chromosome)
    
    logger.debug(f"Generated architecture code: {architecture_code}")
    
    assert architecture_code == expected_architecture_code

def test_copy():
    individual = Individual(max_layers=5)
    individual.architecture = 'LEPELEPEHeadCodeEE'
    individual.chromosome = ['L', 'P', 'L', 'P', 'L', 'P', 'HeadCode']
    individual.fitness = 10.0
    
    logger.debug(f"Original individual: {individual}")
    
    copied_individual = individual.copy()
    
    logger.debug(f"Copied individual: {copied_individual}")
    
    assert copied_individual.architecture == individual.architecture
    assert copied_individual.chromosome == individual.chromosome
    assert copied_individual.fitness == individual.fitness
    assert copied_individual is not individual
    assert copied_individual.chromosome is not individual.chromosome

def test_fitness_initialization():
    individual = Individual(max_layers=5)
    assert individual.fitness == 0.0