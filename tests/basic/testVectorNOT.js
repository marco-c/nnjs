var assert = require('assert');

import * as nnjs from '../../nnjs';

suite('Learn Vector NOT', function() {
  var data = [
    { x: [0, 0], y: [1, 1] },
    { x: [0, 1], y: [1, 0] },
    { x: [1, 0], y: [0, 1] },
    { x: [1, 1], y: [0, 0] },
  ];

  var trainVectors = data.map(val => val.x);
  var trainLabels = data.map(val => val.y);

  function doTest(layers, numIter) {
    var network = new nnjs.Network(layers, 1, 1, 2);

    var trainer = new nnjs.Trainer(network, numIter, 1, 0.1, 0.0005);
    trainer.train(trainVectors, trainLabels);

    var prediction;

    prediction = network.fprop([0, 0]).data;
    assert.equal(Math.round(prediction[0]), 1, "NOT [0, 0] - First");
    assert.equal(Math.round(prediction[1]), 1, "NOT [0, 0] - Second");

    prediction = network.fprop([0, 1]).data;
    assert.equal(Math.round(prediction[0]), 1, "NOT [0, 1] - First");
    assert.equal(Math.round(prediction[1]), 0, "NOT [0, 1] - Second");

    prediction = network.fprop([1, 0]).data;
    assert.equal(Math.round(prediction[0]), 0, "NOT [1, 0] - First");
    assert.equal(Math.round(prediction[1]), 1, "NOT [1, 0] - Second");

    prediction = network.fprop([1, 1]).data;
    assert.equal(Math.round(prediction[0]), 0, "NOT [1, 1] - First");
    assert.equal(Math.round(prediction[1]), 0, "NOT [1, 1] - Second");
  }

  test('Two layer network', function() {
    doTest([
      new nnjs.LinearLayer(2),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(2),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 40000);
  });
});
