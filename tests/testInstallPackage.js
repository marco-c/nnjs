var assert = require('assert');
var childProcess = require('child_process');
var fs = require('fs');
var path = require('path');
var temp = require('temp').track();

function spawn(command, args) {
  return new Promise(function(resolve, reject) {
    var child = childProcess.spawn(command, args);

    child.stdout.on('data', function(chunk) {
      process.stdout.write(chunk);
    });

    child.stderr.on('data', function(chunk) {
      process.stderr.write(chunk);
    });

    child.on('exit', function(code, signal) {
      if (code === 0) {
        resolve(code);
      } else {
        reject(code);
      }
    });

    child.on('error', function(err) {
      reject(err);
    });
  });
}

suite('nnjs package', function() {
  var oldWD = process.cwd();

  beforeEach(function() {
    process.chdir(temp.mkdirSync('nnjs'));
  });

  afterEach(function() {
    process.chdir(oldWD);
  });

  it('should be installable and requireable', function() {
    return spawn('npm', ['install', path.dirname(__dirname)])
    .then(function() {
      fs.writeFileSync('test.js', '\n\
var nnjs = require(\'nnjs\');\n\
if (!nnjs) {\n\
  throw new Error(\'nnjs is not defined\');\n\
}\n\
var symbols = [\'Blob\', \'ConvolutionLayer\', \'PoolingLayer\', \'ReLULayer\', \'SigmoidLayer\',\n\
               \'TanhLayer\', \'LinearLayer\', \'RegressionLayer\', \'SoftmaxLayer\', \'Network\',\n\
               \'Trainer\'];\n\
symbols.forEach(function(symbol) {\n\
  if (!(symbol in nnjs)) {\n\
    throw new Error(symbol + \' is not defined in nnjs\');\n\
  }\n\
});');

      return spawn('node', ['test.js']);
    });
  });
});
