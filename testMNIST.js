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
    new RegressionLayer(10),
  ]);

  var trainer = new Trainer(network, 30 * trainImages.length, 1, 0.1, 0.0005);
  trainer.train(trainImages, trainLabels, testImages, testLabels);

  console.log(network.fprop(testImages[0]));
  console.log(Util.argmax(network.fprop(testImages[0]).data));
  console.log(testLabels[0]);
  console.log(network.fprop(testImages[1]));
  console.log(Util.argmax(network.fprop(testImages[1]).data));
  console.log(testLabels[1]);
  console.log(network.fprop(testImages[2]));
  console.log(Util.argmax(network.fprop(testImages[2]).data));
  console.log(testLabels[2]);
})();
