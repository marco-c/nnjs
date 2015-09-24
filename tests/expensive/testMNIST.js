var assert = require('assert');

import * as nnjs from '../../nnjs';

suite('MNIST', function() {
  this.timeout(0);

  var trainImages = require("../../data/mnist/mnist_train_images.json");
  var testImages = require("../../data/mnist/mnist_test_images.json");
  var trainLabels = require("../../data/mnist/mnist_train_labels.json");
  var testLabels = require("../../data/mnist/mnist_test_labels.json");

  test('Multilayer perceptron', function() {
    var network = new nnjs.Network([
      new nnjs.LinearLayer(784),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(30),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(10),
      new nnjs.SigmoidLayer(),
      new nnjs.SoftmaxLayer(),
    ], 1, 1, 784);

    var trainer = new nnjs.Trainer(network, trainImages.length, 1, 0.1, 0.0005, 0.0, 100);
    trainer.train(trainImages, trainLabels);

    var accuracy = trainer.test(testImages, testLabels);
    assert(accuracy > 0.8, "MNIST classification accuracy (" + (accuracy * 100) + "%) > 80%");
  });

  test('Convolutional neural network', function() {
    var network = new nnjs.Network([
      new nnjs.ConvolutionLayer(8, 5, 1, 0, () => 0.1),
      new nnjs.ReLULayer(),
      new nnjs.PoolingLayer(null, 2, 2, 0),
      new nnjs.ConvolutionLayer(16, 5, 1, 0, () => 0.1),
      new nnjs.ReLULayer(),
      new nnjs.PoolingLayer(null, 2, 2, 0),
      new nnjs.LinearLayer(10, () => 0.0),
      new nnjs.SoftmaxLayer(),
    ], 28, 28, 1);

    var trainer = new nnjs.Trainer(network, trainImages.length, 100, 0.1, 0.001, 0.5, 100);
    trainer.train(trainImages, trainLabels);

    var accuracy = trainer.test(testImages, testLabels);
    assert(accuracy > 0.9, "MNIST classification accuracy (" + (accuracy * 100) + "%) > 90%");
  });
});
