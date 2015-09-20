/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testMNISTWithCNN = (function() {
  load("data/mnist/mnist_train_images.js");
  load("data/mnist/mnist_test_images.js");
  load("data/mnist/mnist_train_labels.js");
  load("data/mnist/mnist_test_labels.js");

  var network = new Network([
    new ConvolutionLayer(8, 5, 1, 0, () => 0.1),
    new ReLULayer(),
    new PoolingLayer(null, 2, 2, 0),
    new ConvolutionLayer(16, 5, 1, 0, () => 0.1),
    new ReLULayer(),
    new PoolingLayer(null, 2, 2, 0),
    new LinearLayer(10, () => 0.0),
    new SoftmaxLayer(),
  ], 28, 28, 1);

  var trainer = new Trainer(network, trainImages.length, 100, 0.1, 0.001, 0.5, 100);
  trainer.train(trainImages, trainLabels);

  var accuracy = trainer.test(testImages, testLabels);
  console.log("Accuracy: " + accuracy);
})();
