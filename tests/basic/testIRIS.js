var assert = require('assert');
var fs = require('fs');

import * as nnjs from '../../nnjs';
import Util from '../../util';

suite('IRIS', function() {
  test('Learn IRIS', function() {
    var classes = {
      "Iris-setosa": 0,
      "Iris-versicolor": 1,
      "Iris-virginica": 2,
    };

    var trainVectors = [];
    var trainLabels = [];

    var irisCSV = fs.readFileSync("data/iris/iris.csv", { encoding: 'utf8' }).split("\n").slice(1,151);
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

    var network = new nnjs.Network([
      new nnjs.LinearLayer(4),
      new nnjs.SigmoidLayer(),
      new nnjs.LinearLayer(3),
      new nnjs.SigmoidLayer(),
      new nnjs.RegressionLayer(),
    ], 1, 1, 4);

    var trainer = new nnjs.Trainer(network, 40 * trainVectors.length, 1, 0.1, 0.0005);
    trainer.train(trainVectors, trainLabels, testVectors, testLabels);

    var accuracy = trainer.test(testVectors, testLabels);
    assert(accuracy > 0.6, "Iris classification accuracy > 60%");
  });
});
