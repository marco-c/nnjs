var assert = require('assert');
var nnjs   = require('../../dist/nnjs');

suite('nodejs module', function() {
  test('is defined', function() {
    assert(nnjs);
  });

  test('has LinearLayer', function() {
    assert(nnjs.LinearLayer);
  });
});
