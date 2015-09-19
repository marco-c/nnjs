/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('rand.js');
load('util.js');

var Blob = function(width, height, depth, num=1) {
  this.width = width;
  this.height = height;
  this.depth = depth;
  this.num = num;

  this.data = new Float64Array(width * height * depth * num);
  this.delta = new Float64Array(width * height * depth * num);
};

var ConvolutionLayer = function(inputWidth, inputHeight, inputDepth, outputDepth, windowSize, stride, pad, biasFiller) {
  this.windowSize = windowSize;
  this.stride = stride;
  this.pad = pad;

  this.outputWidth = Math.floor((inputWidth + 2 * pad - windowSize) / stride + 1);
  this.outputHeight = Math.floor((inputHeight + 2 * pad - windowSize) / stride + 1);
  this.outputDepth = outputDepth;

  this.blobs = new Array(2);
  this.biases = this.blobs[0] = new Blob(1, 1, outputDepth);
  this.weights = this.blobs[1] = new Blob(windowSize, windowSize, inputDepth, outputDepth);

  this.params = new Array(2);
  this.params[0] = {
    weightDecay: 0,
  };
  this.params[1] = {
    weightDecay: 1,
  };

  for (var i = 0; i < outputDepth; i++) {
    if (biasFiller) {
      this.biases.data[i] = biasFiller();
    } else {
      this.biases.data[i] = Rand.randn(0, 1);
    }
  }
  for (var i = 0; i < windowSize * windowSize * inputDepth * outputDepth; i++) {
    this.weights.data[i] = Rand.randn(0, 1.0 / Math.sqrt(windowSize * windowSize * inputDepth));
  }

  this.inputBlob = null;
  this.blob = new Blob(this.outputWidth, this.outputHeight, this.outputDepth);
};

ConvolutionLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;

  for (var d = 0; d < this.outputDepth; d++) {
    for (var startX = -this.pad, outX = 0; outX < this.outputWidth; startX += this.stride, outX++) {
      for (var startY = -this.pad, outY = 0; outY < this.outputHeight; startY += this.stride, outY++) {
        var val = 0;

        for (var x = startX; x < startX + this.windowSize; x++) {
          for (var y = startY; y < startY + this.windowSize; y++) {
            if (x >= 0 && x < inputBlob.width && y >= 0 && y < inputBlob.height) {
              for (var id = 0; id < inputBlob.depth; id++) {
                val += this.weights.data[d * this.weights.depth * this.weights.width * this.weights.height +
                                         id * this.weights.width * this.weights.height +
                                         (x - startX) * this.weights.height +
                                         (y - startY)] *
                       inputBlob.data[id * inputBlob.width * inputBlob.height + x * inputBlob.height + y];
              }
            }
          }
        }

        val += this.biases.data[d];
        this.blob.data[d * this.outputWidth * this.outputHeight + outX * this.outputHeight + outY] = val;
      }
    }
  }

  return this.blob;
};

ConvolutionLayer.prototype.bprop = function(nextBlob) {
  this.inputBlob.delta.fill(0);

  for (var d = 0; d < this.outputDepth; d++) {
    for (var startX = -this.pad, outX = 0; outX < this.outputWidth; startX += this.stride, outX++) {
      for (var startY = -this.pad, outY = 0; outY < this.outputHeight; startY += this.stride, outY++) {
        var outputIdx = d * this.outputWidth * this.outputHeight + outX * this.outputHeight + outY;

        for (var x = startX; x < startX + this.windowSize; x++) {
          for (var y = startY; y < startY + this.windowSize; y++) {
            if (x >= 0 && x < this.inputBlob.width && y >= 0 && y < this.inputBlob.height) {
              for (var id = 0; id < this.inputBlob.depth; id++) {
                var weightIdx = d * this.weights.depth * this.weights.width * this.weights.height +
                                id * this.weights.width * this.weights.height +
                                (x - startX) * this.weights.height +
                                (y - startY);
                var inputIdx = id * this.inputBlob.width * this.inputBlob.height + x * this.inputBlob.height + y;


                this.weights.delta[weightIdx] += this.inputBlob.data[inputIdx] * nextBlob.delta[outputIdx];
                this.inputBlob.delta[inputIdx] += this.weights.data[weightIdx] * nextBlob.delta[outputIdx];
              }
            }
          }
        }

        this.biases.delta[outputIdx] += nextBlob.delta[outputIdx];
      }
    }
  }

  return this.inputBlob;
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

var LinearLayer = function(numInput, numOutput, biasFiller) {
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
    if (biasFiller) {
      this.biases.data[i] = biasFiller();
    } else {
      this.biases.data[i] = Rand.randn(0, 1);
    }
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
  this.inputBlob.delta.fill(0);

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

  this.loss = 0;
}

RegressionLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;
  return inputBlob;
}

RegressionLayer.prototype.bprop = function(y) {
  this.loss = 0;

  if (Array.isArray(y)) {
    for (var i = 0; i < this.numOutput; i++) {
      this.inputBlob.delta[i] = this.inputBlob.data[i] - y[i];
      this.loss += 0.5 * this.inputBlob.delta[i] * this.inputBlob.delta[i];
    }
  } else {
    this.inputBlob.delta[0] = this.inputBlob.data[0] - y;
    this.loss += 0.5 * this.inputBlob.delta[0] * this.inputBlob.delta[0];
  }

  return this.inputBlob;
}

var SoftmaxLayer = function(numInput) {
  this.numInput = numInput;
  this.numOutput = numInput;

  this.inputBlob = null;
  this.blob = new Blob(1, 1, numInput);

  this.loss = 0;
}

SoftmaxLayer.prototype.fprop = function(inputBlob) {
  this.inputBlob = inputBlob;

  var max = inputBlob.data[0];
  for (var i = 1; i < this.numInput; i++) {
    if (max < inputBlob.data[i]) {
      max = inputBlob.data[i];
    }
  }

  var expSum = 0;
  for (var i = 0; i < this.numInput; i++) {
    this.blob.data[i] = Math.exp(inputBlob.data[i] - max);
    expSum += this.blob.data[i];
  }

  for (var i = 0; i < this.numInput; i++) {
    this.blob.data[i] = this.blob.data[i] / expSum;
  }

  return this.blob;
}

SoftmaxLayer.prototype.bprop = function(y) {
  for (var i = 0; i < this.numOutput; i++) {
    this.inputBlob.delta[i] = this.blob.data[i] - (i === y ? 1.0 : 0.0);
  }

  this.loss = -Math.log(this.blob.data[y]);

  return this.inputBlob;
}

function Network(layers) {
  this.layers = layers;
}

Network.prototype.fprop = function(input) {
  /*var x = new Blob(1, 1, input.length);
  for (var i = 0; i < input.length; i++) {
    x.data[i] = input[i];
  }*/
  var x = new Blob(28, 28, 1);
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
  return grad;
}

function Trainer(network, iterations, batchSize, learningRate, weightDecay=0.0, momentum=null, displayIterations=0, testIterations=0) {
  this.network = network;
  this.iterations = iterations;
  this.batchSize = batchSize;
  this.learningRate = learningRate;
  this.weightDecay = weightDecay;
  this.displayIterations = displayIterations;
  this.testIterations = testIterations;
  this.momentum = momentum;

  if (this.momentum) {
    this.v = new Array(this.network.layers.length);

    for (var i = 0; i < this.network.layers.length; i++) {
      var layer = this.network.layers[i];
      if (layer.blobs) {
        this.v[i] = new Array(layer.blobs.length);

        for (var j = 0; j < layer.blobs.length; j++) {
          var blob = layer.blobs[j];

          this.v[i][j] = new Blob(1, 1, blob.delta.length);
        }
      }
    }
  }
}

Trainer.prototype.train = function(trainVectors, trainLabels, testVectors, testLabels) {
  var inputLen = trainVectors[0].length;

  var lossArr = new Array(this.displayIterations);

  for (var n = 0; n < this.iterations; n++) {
    if (this.displayIterations && n % this.displayIterations === 0) {
      console.log("n: " + n);
      console.log("Loss: " + lossArr.reduce((prev, cur) => prev + cur, 0) / lossArr.length);
    }

    var dataIdx = n % trainVectors.length;

    if (this.testIterations && n % this.testIterations === 0 && testVectors && testLabels) {
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
              var grad = (blob.delta[k] + param.weightDecay * this.weightDecay * blob.data[k]) / this.batchSize;
              var upd = - this.learningRate * grad;
              if (this.momentum) {
                upd += this.momentum * this.v[i][j].data[k];
                this.v[i][j].data[k] = upd;
              }
              blob.data[k] += upd;
            }

            blob.delta.fill(0);
          }
        }
      }
    }

    lossArr[n % this.displayIterations] = this.network.layers[this.network.layers.length - 1].loss;
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
