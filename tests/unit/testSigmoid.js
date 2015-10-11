var assert = require('assert');

import * as nnjs from '../../nnjs';

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

function doTest(inputWidth, inputHeight, inputDepth) {
  var sigmoidLayer = new nnjs.SigmoidLayer;
  sigmoidLayer.init(inputWidth, inputHeight, inputDepth);

  var inputBlob = new nnjs.Blob(inputWidth, inputHeight, inputDepth);
  for (var d = 0; d < inputDepth; d++) {
    for (var x = 0; x < inputWidth; x++) {
      for (var y = 0; y < inputHeight; y++) {
        inputBlob.data[d * inputWidth * inputHeight + x * inputHeight + y] = Math.random();
      }
    }
  }

  for (var d = 0; d < inputDepth; d++) {
    for (var x = 0; x < inputWidth; x++) {
      for (var y = 0; y < inputHeight; y++) {
        var i = d * inputWidth * inputHeight + x * inputHeight + y;

        var result = sigmoidLayer.fprop(inputBlob);
        assert.equal(result.data[i], sigmoid(inputBlob.data[i]),
                     "Correct result (" +  x + "," + y + "," + d + ")");
      }
    }
  }

  var nextBlob = new nnjs.Blob(inputWidth, inputHeight, inputDepth);
  for (var d = 0; d < inputDepth; d++) {
    for (var x = 0; x < inputWidth; x++) {
      for (var y = 0; y < inputHeight; y++) {
        var i = d * inputWidth * inputHeight + x * inputHeight + y;
        nextBlob.delta[i] = i;
      }
    }
  }

  for (var d = 0; d < inputDepth; d++) {
    for (var x = 0; x < inputWidth; x++) {
      for (var y = 0; y < inputHeight; y++) {
        var i = d * inputWidth * inputHeight + x * inputHeight + y;

        var result = sigmoidLayer.bprop(nextBlob);
        assert.equal(result.delta[i],
                     i * sigmoid(inputBlob.data[i]) * (1.0 - sigmoid(inputBlob.data[i])),
                     "Correct bprop result (" +  x + "," + y + "," + d + ")");
      }
    }
  }
}

suite('SigmoidLayer', function() {
  test('create', function() {
    var sigmoidLayer = new nnjs.SigmoidLayer();
    assert(!!sigmoidLayer);
  });

  test('init', function() {
    var sigmoidLayer = new nnjs.SigmoidLayer();
    sigmoidLayer.init(3, 4, 5);
    assert(sigmoidLayer.outputWidth, 3);
    assert(sigmoidLayer.outputHeight, 4);
    assert(sigmoidLayer.outputDepth, 5);
  })

  test('fprop/brop single input', () => doTest(1, 1, 1));

  test('fprop/brop multiple inputs', () => doTest(2, 1, 1));
});
