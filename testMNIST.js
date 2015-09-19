/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testMNIST = (function() {
  load("data/mnist/mnist_train_images.js");
  load("data/mnist/mnist_test_images.js");
  load("data/mnist/mnist_train_labels.js");
  load("data/mnist/mnist_test_labels.js");

  var network = new Network([
    new LinearLayer(784, 784),
    new SigmoidLayer(784),
    new LinearLayer(784, 30),
    new SigmoidLayer(30),
    new LinearLayer(30, 10),
    new SigmoidLayer(10),
    new SoftmaxLayer(10),
  ]);

  var trainer = new Trainer(network, trainImages.length, 1, 0.1, 0.0005, 0.0, 100);
  trainer.train(trainImages, trainLabels);

  var accuracy = trainer.test(testImages, testLabels);
  console.log("Accuracy: " + accuracy);
})();
