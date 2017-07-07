import unittest
import SophosNet as sn
import SophosGauss as sg
import numpy as np

class MainTests(unittest.TestCase):

    def setUp(self):
        self.model_gauss = sg.Model()

    def test_LinearClassification(self):
        
        # Build Model
        model = sn.Model()
        l1 = sn.Layer(2, 1)
        activation1 = sn.Activation('sigmoid')
        model.add(l1)
        l1.setWeights(np.matrix('.5; .5; .5'))
        model.add(activation1)
        model.setLearningRate(0.1)

        # Setup data and train
        x_input = np.matrix('0 0')
        y_input = np.matrix('0')
        model.train(x_input, y_input)

        # First Weights
        weights_first = l1.getWeights()

        x_input = np.matrix('1 1')
        y_input = np.matrix('1')
        model.train(x_input, y_input)

        # Set ending weights
        weights_end = l1.getWeights()


        # Set expected weights
        weights_first_expected = np.matrix('4853; 5000; 5000')
        weights_end_expected = np.matrix('4881; 5027; 5027')
        weights_first = (weights_first * 10000).astype(int)
        weights_end = (weights_end * 10000).astype(int)
        t1 = False
        t2 = False
        if (weights_first == weights_first_expected).all():
            t1 = True
        if (weights_end == weights_end_expected).all():
            t2 = True

        # Equality Tests
        self.assertTrue(t1)
        self.assertTrue(t2)

    def test_BackpropagationMultiLayerBatched(self):
        model = sn.Model()
        x_input = np.matrix('0 1; 1 0')
        y = np.matrix('0; 1')
        l1 = sn.Layer(2, 1)
        # l2 = sn.Layer(2, 1)
        # l1.setWeights(w1)
        # l2.setWeights(w2)
        activation1 = sn.Activation('sigmoid')
        # activation2 = sn.Activation('sigmoid')
        model.add(l1)
        model.add(activation1)
        # model.add(l2)
        # model.add(activation2)
        original_output = model.feed(x_input)
        
        for i in range(1000):
            model.train_batch(x_input, y)
            if i%100 == 0:
                print("Error: ", model.getTotalError())
        print("Weights: ", l1.getWeights())    
        print("Original Output: ", original_output)
        print("Trained Output: ", model.feed(np.matrix('0 1')))


    def test_BackpropagationMultiLayer(self):
        model = sn.Model()
        x_input = np.matrix('.05 .1')
        y = np.matrix('.01 .99')
        w1 = np.matrix('.35 .35; .15 .25; .20 .30')
        w2 = np.matrix('.6 .6; .4 .5; .45 .55')
        l1 = sn.Layer(2, 2)
        l2 = sn.Layer(2, 2)
        l1.setWeights(w1)
        l2.setWeights(w2)
        activation1 = sn.Activation('sigmoid')
        activation2 = sn.Activation('sigmoid')
        model.add(l1)
        model.add(activation1)
        model.add(l2)
        model.add(activation2)
        original_output = model.feed(x_input)
        model.train(x_input, y)
        updated_weights = l2.getWeights()
        updated_weights_l1 = l1.getWeights()
        expected_weights = np.matrix('.53075072 0.61904912; 0.35891648 0.51130127; 0.40866619 0.56137012')
        expected_weights_l1 = np.matrix('0.34561432 0.34502287; 0.14978072 0.24975114; 0.19956143 0.29950229')
        self.assertEqual(updated_weights_l1.all(), expected_weights_l1.all())
        self.assertEqual(updated_weights.all(), expected_weights.all())

    def test_backpropagationSingleLayer(self):

        pass

    def test_TotalError(self):
        model = sn.Model()
        x_input = np.matrix('.05 .1')
        y = np.matrix('.01 .99')
        w1 = np.matrix('.35 .35; .15 .25; .20 .30')
        w2 = np.matrix('.6 .6; .4 .5; .45 .55')
        l1 = sn.Layer(2, 2)
        l2 = sn.Layer(2, 2)
        l1.setWeights(w1)
        l2.setWeights(w2)
        activation = sn.Activation('sigmoid')
        model.add(l1)
        model.add(activation)
        model.add(l2)
        model.add(activation)
        model.train(x_input, y)
        # Value confirmed from online source and by hand
        self.assertEqual(model.getTotalError(), 0.29837110876000272)

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