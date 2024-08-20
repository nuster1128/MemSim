"""
@Name: main.py
@Author: Zeyu Zhang
@Date: 2024/4/28-15:37

Script: This is the main program for memory evaluation.
"""

from Evaluator import Evaluator
from Display import Display
from utils import load_config
import numpy as np
import random

np.random.seed(1128)
random.seed(1128)


def run(config):
    # Create evaluator
    evaluator = Evaluator(config)
    # Get result
    evaluator.eval()
    result = evaluator.get_result()
    # Show result
    display = Display(config,result)
    display.table_show()

if __name__ == '__main__':
    config = load_config('configs/glm4local.yaml')
    run(config)