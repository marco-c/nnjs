var assert = require('assert');

import * as nnjs from '../../nnjs';

function relu(x) {
  return x >= 0 ? x : 0;
}

function doTest(inputWidth, inputHeight, inputDepth) {
  var reLULayer = new nnjs.ReLULayer;
  reLULayer.init(inputWidth, inputHeight, inputDepth);

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

        var result = reLULayer.fprop(inputBlob);
        assert.equal(result.data[i], relu(inputBlob.data[i]),
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

        var result = reLULayer.bprop(nextBlob);
        assert.equal(result.delta[i],
                     relu(inputBlob.data[i]) > 0 ? i : 0,
                     "Correct bprop result (" +  x + "," + y + "," + d + ")");
      }
    }
  }
}

suite('ReLULayer', function() {
  test('create', function() {
    var reLULayer = new nnjs.ReLULayer();
    assert(!!reLULayer);
  });

  test('init', function() {
    var reLULayer = new nnjs.ReLULayer();
    reLULayer.init(3, 4, 5);
    assert(reLULayer.outputWidth, 3);
    assert(reLULayer.outputHeight, 4);
    assert(reLULayer.outputDepth, 5);
  })

  test('fprop/brop single input', () => doTest(1, 1, 1));

  test('fprop/brop multiple inputs', () => doTest(2, 1, 1));
});
