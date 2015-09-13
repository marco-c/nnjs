/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testIRIS = (function() {
  var classes = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
  };

  var trainVectors = [];
  var trainLabels = [];

  var irisCSV = snarf("data/iris/iris.csv").split("\n").slice(1,151);
  Util.shuffle(irisCSV);
  var irisTrainingSet = irisCSV.slice(0,120);
  var irisTestSet = irisCSV.slice(120);
  for (var i = 0; i < irisTrainingSet.length; i++) {
    var elems = irisTrainingSet[i].split(",");

    trainVectors.push(elems.slice(0,4).map(function(elem) {
      return Number(elem);
    }));

    var y = new Array(3).fill(0);
    y[classes[elems[4]]] = 1;
    trainLabels.push(y);
  }

  var testVectors = [];
  var testLabels = [];
  for (var i = 0; i < irisTestSet.length; i++) {
    var elems = irisTestSet[i].split(",");

    testVectors.push(elems.slice(0,4).map(function(elem) {
      return Number(elem);
    }));

    testLabels.push(classes[elems[4]]);
  }

  var network = new Network([
    new LinearLayer(4, 4),
    new SigmoidLayer(4),
    new LinearLayer(4, 3),
    new SigmoidLayer(3),
    new RegressionLayer(3),
  ]);

  var trainer = new Trainer(network, 40 * trainVectors.length, 1, 0.1, 0.0005);
  trainer.train(trainVectors, trainLabels, testVectors, testLabels);

  var accuracy = trainer.test(testVectors, testLabels);
  console.log("Iris classification with " + accuracy + " accuracy");
})();
