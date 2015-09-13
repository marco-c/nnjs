/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testAND = (function() {
  var data = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 0 },
    { x: [1, 0], y: 0 },
    { x: [1, 1], y: 1 },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  var network = new Network([
    new LinearLayer(2, 1),
    new SigmoidLayer(1),
    new RegressionLayer(1),
  ]);

  var trainer = new Trainer(network, 4000, 1, 0.1);
  trainer.train(trainVectors, trainLabels);

  if (Math.round(network.fprop([0, 0]).data[0]) !== 0) {
    console.log("FAIL - 0 AND 0");
  }

  if (Math.round(network.fprop([0, 1]).data[0]) !== 0) {
    console.log("FAIL - 0 AND 1");
  }

  if (Math.round(network.fprop([1, 0]).data[0]) !== 0) {
    console.log("FAIL - 1 AND 0");
  }

  if (Math.round(network.fprop([1, 1]).data[0]) !== 1) {
    console.log("FAIL - 1 AND 1");
  }
})();