/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testNOTWithReLU = (function() {
  var data = [
    { x: [0], y: 1 },
    { x: [1], y: 0 },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  var network = new Network([
    new LinearLayer(1),
    new ReLULayer(),
    new RegressionLayer(),
  ], 1, 1, 1);

  var trainer = new Trainer(network, 2000, 1, 0.1);
  trainer.train(trainVectors, trainLabels);

  if (Math.round(network.fprop([0]).data[0]) !== 1) {
    console.log("FAIL - NOTWithReLU 0");
  }

  if (Math.round(network.fprop([1]).data[0]) !== 0) {
    console.log("FAIL - NOTWithReLU 1");
  }
})();
