import unittest
import SophosNet as sn
import SophosGauss as sg
import numpy as np

class MainTests(unittest.TestCase):

	def setUp(self):
		self.model_gauss = sg.Model()
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

	def test_ForwardProp(self):
		self.model = sn.Model()
		l1 = sn.Layer(2, 2)
		l2 = sn.Layer(2, 2)
		l3 = sn.Layer(2, 1)
		self.model.add(l1)
		self.model.add(l2)
		self.model.add(l3)
		weights = np.matrix('.5 .5; .5 .5; .5 .5')
		l1.setWeights(weights)
		l2.setWeights(weights)
		l3.setWeights(np.matrix('.5; .5; .5'))
		feed_x = np.matrix(('1 1'))
		result = self.model.feed(feed_x)
		# print(self.model.feed(feed_x))
		self.assertEqual(result, 2.5)

	def test_FeedLayer(self):
		i = np.matrix(('1, 3, 3'))
		input_with_bias = np.matrix(('1, 1, 3, 3'))
		l1 = sn.Layer(3,2)
		l1_output = l1.feed(i)
		expected_result = input_with_bias * l1.getWeights()
		testValue = np.array_equal(l1_output, expected_result)
		self.assertTrue(testValue)

	def test_GaussModel(self):
		# Mean and covariance
		mu = np.matrix('0 0')
		sigma = np.matrix('1 0; 0 1')

		# Input point
		x = np.matrix('2 2')
		p1 = self.model_gauss.predictGaussian(mu, sigma, x)
		output = p1.item(0,0)
		self.assertEqual(output, 0.0029150244650281935)

	def test_GaussMuClass(self):
		input_data = np.matrix('1 1; 2 2; 3 3; 5 5')
		expectedResult = np.matrix('2.75 2.75')
		result = self.model_gauss.mu_class(input_data)
		testValue = np.array_equal(expectedResult, result)
		self.assertTrue(testValue)
		pass

	def test_LayerShape(self):
		baseShape = [2, 3]
		expectedShape = [3, 3]
		L1 = sn.Layer(baseShape[0], baseShape[1])
		layerShape = L1.getShape()

		self.assertEqual(expectedShape, layerShape)

if __name__ == '__main__':
	unittest.main()