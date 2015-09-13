/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testVectorNOT = (function() {
  var data = [
    { x: [0, 0], y: [1, 1] },
    { x: [0, 1], y: [1, 0] },
    { x: [1, 0], y: [0, 1] },
    { x: [1, 1], y: [0, 0] },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  var network = new Network([
    new LinearLayer(2, 2),
    new SigmoidLayer(2),
    new LinearLayer(2, 2),
    new SigmoidLayer(2),
    new RegressionLayer(2),
  ]);

  var trainer = new Trainer(network, 40000, 1, 0.1, 0.0005);
  trainer.train(trainVectors, trainLabels);

  var prediction = network.fprop([0, 0]).data;
  if (Math.round(prediction[0]) !== 1 && Math.round(prediction[1]) !== 1) {
    console.log("FAIL - NOT [0, 0]");
  }

  prediction = network.fprop([0, 1]).data;
  if (Math.round(prediction[0]) !== 1 && Math.round(prediction[1]) !== 0) {
    console.log("FAIL - NOT [0, 1]");
  }

  prediction = network.fprop([1, 0]).data;
  if (Math.round(prediction[0]) !== 0 && Math.round(prediction[1]) !== 1) {
    console.log("FAIL - NOT [1, 0]");
  }

  prediction = network.fprop([1, 1]).data;
  if (Math.round(prediction[0]) !== 0 && Math.round(prediction[1]) !== 0) {
    console.log("FAIL - NOT [1, 1]");
  }
})();
