import unittest
import SophosNet as sn
import numpy as np

class MainTests(unittest.TestCase):

	def setUp(self):
		self.model = sn.Model()
		# i = np.matrix(('1, 3'))
		# # Define Model Layers
		# L1 = sn.Layer(3, 2)
		# model.add(L1)
		# activation1 = Activation('sigmoid')
		# model.add(activation1)
		# model.add(Layer(1, 3))
		# model.add(Activation('sigmoid'))
		# model.train(i, 1)
		# model.show()
		pass

	def test_FeedLayer(self):
		i = np.matrix(('1, 3'))
		

	def test_LayerShape(self):
		baseShape = [2, 3]
		expectedShape = [3, 3]
		L1 = sn.Layer(baseShape[0], baseShape[1])
		layerShape = L1.getShape()

		self.assertEqual(expectedShape, layerShape)

if __name__ == '__main__':
	unittest.main()