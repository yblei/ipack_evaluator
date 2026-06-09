import unittest

import numpy as np

import evaluate_example
from c_evaluator.grocery_item import GroceryItem


class TestEvaluateExample(unittest.TestCase):
	def test_example_inference_assigns_expected_obbs(self):
		grocery_items = [
			GroceryItem(name=f"item-{index}", extent=np.array([0.1, 0.2, 0.3]))
			for index in range(3)
		]

		returned_items = evaluate_example.test_inference(
			grocery_items=grocery_items,
			boxes={"test-box": np.array([0.5, 0.5, 0.5])},
			visualizer=None,
		)

		self.assertIs(returned_items, grocery_items)

		for index, item in enumerate(returned_items):
			self.assertIsNotNone(item.obb_target)
			np.testing.assert_allclose(
				item.obb_target.center,
				np.array([0.3, 0.3, 0.1 * index]),
			)
			np.testing.assert_allclose(item.obb_target.extent, item.extent)
			np.testing.assert_allclose(item.obb_target.R, np.eye(3))