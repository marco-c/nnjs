var assert = require('assert');

import * as nnjs from '../../nnjs';

function doTest(inputWidth, inputHeight, inputDepth) {
  var tanhLayer = new nnjs.TanhLayer;
  tanhLayer.init(inputWidth, inputHeight, inputDepth);

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

        var result = tanhLayer.fprop(inputBlob);
        assert.equal(result.data[i], Math.tanh(inputBlob.data[i]),
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

        var result = tanhLayer.bprop(nextBlob);
        assert.equal(result.delta[i],
                     i * (1.0 - Math.tanh(inputBlob.data[i]) * Math.tanh(inputBlob.data[i])),
                     "Correct bprop result (" +  x + "," + y + "," + d + ")");
      }
    }
  }
}

suite('TanhLayer', function() {
  test('create', function() {
    var tanhLayer = new nnjs.TanhLayer();
    assert(!!tanhLayer);
  });

  test('init', function() {
    var tanhLayer = new nnjs.TanhLayer();
    tanhLayer.init(3, 4, 5);
    assert(tanhLayer.outputWidth, 3);
    assert(tanhLayer.outputHeight, 4);
    assert(tanhLayer.outputDepth, 5);
  })

  test('fprop/brop single input', () => doTest(1, 1, 1));

  test('fprop/brop multiple inputs', () => doTest(2, 1, 1));
});
