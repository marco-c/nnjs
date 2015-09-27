var assert = require('assert');

import * as nnjs from '../../nnjs';

suite('Learn NOT', function() {
  var data = [
    { x: [0], y: 1 },
    { x: [1], y: 0 },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  function doTest(layers, numIter) {
    var network = new nnjs.Network(layers, 1, 1, 1);

    var trainer = new nnjs.Trainer(network, numIter, 1, 0.1);
    trainer.train(trainVectors, trainLabels);

    assert.equal(Math.round(network.fprop([0]).data[0]), 1, "NOT 0");
    assert.equal(Math.round(network.fprop([1]).data[0]), 0, "NOT 1");
  }

  test('Single layer network', function() {
    doTest([
      new nnjs.LinearLayer(1),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 1000);
  });

  test('Single layer network with ReLU', function() {
    doTest([
      new nnjs.LinearLayer(1, () => 0.1),
      new nnjs.ReLULayer(),
      new nnjs.RegressionLayer(),
    ], 2000);
  });

  test('Single layer network with tanh', function() {
    doTest([
      new nnjs.LinearLayer(1),
      new nnjs.TanhLayer(),
      new nnjs.RegressionLayer(),
    ], 1000);
  });

  test('Two layer network', function() {
    doTest([
      new nnjs.LinearLayer(1),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(1),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 10000);
  });
});
