var assert = require('assert');

import * as nnjs from '../../nnjs';

suite('Learn XOR', function() {
  var data = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  function doTest(layers, numIter, momentum) {
    var network = new nnjs.Network(layers, 1, 1, 2);

    var trainer = new nnjs.Trainer(network, numIter, 1, 0.1, 0.0001, momentum);
    trainer.train(trainVectors, trainLabels);

    assert.equal(Math.round(network.fprop([0, 0]).data[0]), 0, "0 XOR 0");
    assert.equal(Math.round(network.fprop([0, 1]).data[0]), 1, "0 XOR 1");
    assert.equal(Math.round(network.fprop([1, 0]).data[0]), 1, "1 XOR 0");
    assert.equal(Math.round(network.fprop([1, 1]).data[0]), 0, "1 XOR 1");
  }

  test('Two layer network', function() {
    doTest([
      new nnjs.LinearLayer(2),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(1),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 80000);
  });

  test('Two layer network with momentum', function() {
    doTest([
      new nnjs.LinearLayer(2),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(1),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 40000, 0.9);
  });
});
