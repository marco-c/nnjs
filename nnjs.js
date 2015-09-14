/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('rand.js');
load('util.js');

var Blob = function(width, height, depth) {
  this.width = width;
  this.height = height;
  this.depth = depth;

  this.data = new Float64Array(width * height * depth);
  this.delta = new Float64Array(width * height * depth);
};

var ConvolutionLayer = function(inputWidth, inputHeight, inputDepth, windowSize, stride, pad) {
  this.numInput = numInput;
  this.numOutput = numOutput;
  this.windowSize = windowSize;
  this.stride = stride;
  this.pad = pad;

  this.outputWidth = Math.floor((inputWidth + 2 * pad - windowSize) / stride + 1);
  this.outputHeight = Math.floor((inputHeight + 2 * pad - windowSize) / stride + 1);
  this.outputDepth = inputDepth;

  this.blobs = new Array(2);
  this.biases = this.blobs[0] = new Blob(1, 1, numOutput);
  this.weights = this.blobs[1] = new Blob(windowSize, windowSize, inputDepth);

  this.params = new Array(2);
  this.params[0] = {
    weightDecay: 0,
  };
  this.params[1] = {
    weightDecay: 1,
  };

  for (var i = 0; i < numOutput; i++) {
    this.biases.data[i] = Rand.randn(0, 1);
  }
  for (var i = 0; i < numOutput * numInput; i++) {
    this.weights.data[i] = Rand.randn(0, 1.0 / Math.sqrt(numInput));
  }
};

ConvolutionLayer.prototype.fprop = function() {

};

ConvolutionLayer.prototype.bprop = function() {

};

// XXX: Only MaxPooling for now.
var PoolingLayer = function(inputWidth, inputHeight, inputDepth, poolingFunc, windowSize, stride, pad) {
  this.poolingFunc = poolingFunc;
  this.windowSize = windowSize;
  this.stride = stride;
  this.pad = pad;

  this.outputWidth = Math.floor((inputWidth + 2 * pad - windowSize) / stride + 1);
  this.outputHeight = Math.floor((inputHeight + 2 * pad - windowSize) / stride + 1);
  this.outputDepth = inputDepth;

  this.inputBlob = null;
  this.blob = new Blob(this.outputWidth, this.outputHeight, this.outputDepth);

  this.gradMap = Object.create(null);
}

PoolingLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;

  for (var d = 0; d < inputBlob.depth; d++) {
    for (var startX = -this.pad, outX = 0; outX < this.outputWidth; startX += this.stride, outX++) {
      for (var startY = -this.pad, outY = 0; outY < this.outputHeight; startY += this.stride, outY++) {
        var maxIdx = -1;
        var max = Number.NEGATIVE_INFINITY;

        for (var x = startX; x < startX + this.windowSize; x++) {
          for (var y = startY; y < startY + this.windowSize; y++) {
            if (x >= 0 && x < inputBlob.width && y >= 0 && y < inputBlob.height) {
              var idx = d * inputBlob.width * inputBlob.height + x * inputBlob.height + y;
              var val = inputBlob.data[idx];
              if (val > max) {
                max = val;
                maxIdx = idx;
              }
            }
          }
        }

        var outIdx = d * this.outputWidth * this.outputHeight + outX * this.outputHeight + outY;
        this.blob.data[outIdx] = max;
        this.gradMap[outIdx] = maxIdx;
      }
    }
  }

  return this.blob;
};

PoolingLayer.prototype.bprop = function(nextBlob) {
  this.inputBlob.delta.fill(0);

  for (var i = 0; i < nextBlob.depth * nextBlob.width * nextBlob.height; i++) {
    this.inputBlob.delta[this.gradMap[i]] += nextBlob.delta[i];
    delete this.gradMap[i];
  }

  return this.inputBlob;
};

var ReLULayer = function(numInput) {
  this.numInput = numInput;
  this.numOutput = numInput;

  this.blob = new Blob(1, 1, numInput);
};

ReLULayer.prototype.fprop = function(inputBlob) {
  for (var i = 0; i < this.numInput; i++) {
    this.blob.data[i] = inputBlob.data[i] >= 0 ? inputBlob.data[i] : 0;
  }
  return this.blob;
};

ReLULayer.prototype.bprop = function(nextBlob) {
  for (var i = 0; i < this.numInput; i++) {
    this.blob.delta[i] = this.blob.data[i] > 0 ? nextBlob.delta[i] : 0;
  }
  return this.blob;
};

var SigmoidLayer = function(numInput) {
  this.numInput = numInput;
  this.numOutput = numInput;

  this.blob = new Blob(1, 1, numInput);
};

SigmoidLayer.prototype.fprop = function(inputBlob) {
  for (var i = 0; i < this.numInput; i++) {
    this.blob.data[i] = 1.0 / (1.0 + Math.exp(-inputBlob.data[i]));
  }
  return this.blob;
};

SigmoidLayer.prototype.bprop = function(nextBlob) {
  for (var i = 0; i < this.numInput; i++) {
    this.blob.delta[i] = this.blob.data[i] * (1.0 - this.blob.data[i]) * nextBlob.delta[i];
  }
  return this.blob;
};

var LinearLayer = function(numInput, numOutput) {
  this.numInput = numInput;
  this.numOutput = numOutput;

  this.blobs = new Array(2);
  this.biases = this.blobs[0] = new Blob(1, 1, numOutput);
  this.weights = this.blobs[1] = new Blob(1, 1, numOutput * numInput);

  this.params = new Array(2);
  this.params[0] = {
    weightDecay: 0,
  };
  this.params[1] = {
    weightDecay: 1,
  };

  for (var i = 0; i < numOutput; i++) {
    this.biases.data[i] = Rand.randn(0, 1);
  }
  for (var i = 0; i < numOutput * numInput; i++) {
    this.weights.data[i] = Rand.randn(0, 1.0 / Math.sqrt(numInput));
  }

  this.inputBlob = null;

  this.blob = new Blob(1, 1, numOutput);
};

LinearLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;

  for (var i = 0; i < this.numOutput; i++) {
    this.blob.data[i] = this.biases.data[i];

    for (var j = 0; j < this.numInput; j++) {
      this.blob.data[i] += this.weights.data[i * this.numOutput + j] * inputBlob.data[j];
    }
  }

  return this.blob;
}

LinearLayer.prototype.bprop = function(nextBlob) {
  for (var i = 0; i < this.numOutput; i++) {
    this.biases.delta[i] += nextBlob.delta[i];
  }

  for (var i = 0; i < this.numOutput; i++) {
    for (var j = 0; j < this.numInput; j++) {
      this.inputBlob.delta[j] += this.weights.data[i * this.numOutput + j] * nextBlob.delta[i];
      this.weights.delta[i * this.numOutput + j] += this.inputBlob.data[j] * nextBlob.delta[i];
    }
  }

  return this.inputBlob;
}

var RegressionLayer = function(numInput) {
  this.numInput = numInput;
  this.numOutput = numInput;

  this.inputBlob = null;
}

RegressionLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;
  return inputBlob;
}

RegressionLayer.prototype.bprop = function(y) {
  if (Array.isArray(y)) {
    for (var i = 0; i < this.numOutput; i++) {
      this.inputBlob.delta[i] = this.inputBlob.data[i] - y[i];
    }
  } else {
    this.inputBlob.delta[0] = this.inputBlob.data[0] - y;
  }

  return this.inputBlob;
}

function Network(layers) {
  this.layers = layers;
}

Network.prototype.fprop = function(input) {
  var x = new Blob(1, 1, input.length);
  for (var i = 0; i < input.length; i++) {
    x.data[i] = input[i];
  }

  for (var i = 0; i < this.layers.length; i++) {
    x = this.layers[i].fprop(x);
  }
  return x;
}

Network.prototype.bprop = function(y) {
  var grad = y;
  for (var i = this.layers.length - 1; i >= 0; i--) {
    grad = this.layers[i].bprop(grad);
  }
}

function Trainer(network, iterations, batchSize, learningRate, weightDecay) {
  this.network = network;
  this.iterations = iterations;
  this.batchSize = batchSize;
  this.learningRate = learningRate;
  this.weightDecay = weightDecay || 0.0;
}

Trainer.prototype.train = function(trainVectors, trainLabels, testVectors, testLabels) {
  var inputLen = trainVectors[0].length;

  for (var n = 0; n < this.iterations; n++) {
    /*if (n % 500 === 0) {
      console.log("n: " + n);
    }*/

    var dataIdx = n % trainVectors.length;

    if (dataIdx === 0 && testVectors && testLabels) {
      console.log("Iter " + n + ": " + (this.test(testVectors, testLabels) * 100) + "%");
    }

    var x = trainVectors[dataIdx];
    var correctY = trainLabels[dataIdx];

    this.network.fprop(x);
    this.network.bprop(correctY);

    if (n % this.batchSize === 0) {
      for (var i = 0; i < this.network.layers.length; i++) {
        var layer = this.network.layers[i];
        if (layer.blobs) {
          for (var j = 0; j < layer.blobs.length; j++) {
            var blob = layer.blobs[j];
            var param = layer.params[j];

            for (var k = 0; k < blob.delta.length; k++) {
              var grad = blob.delta[k] + param.weightDecay * this.weightDecay * blob.data[k];
              blob.data[k] -= this.learningRate * grad;
            }

            blob.delta.fill(0);
          }
        }
      }
    }
  }
};

// XXX: This only works for classification problems.
Trainer.prototype.test = function(testVectors, testLabels) {
  var correct = 0;

  for (var i = 0; i < testVectors.length; i++) {
    if (Util.argmax(this.network.fprop(testVectors[i]).data) === testLabels[i]) {
      correct++;
    }
  }

  return correct / testVectors.length;
};
