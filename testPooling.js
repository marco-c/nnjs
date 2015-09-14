/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

load('nnjs.js');

var testPooling = (function() {
  function test(name, windowSize, stride, pad, data, expected, expectedDelta) {
    var depth = data.length;
    var width = data[0].length;
    var height = data[0][0].length;

    var blob = new Blob(width, height, depth);
    for (var d = 0; d < depth; d++) {
      for (var x = 0; x < width; x++) {
        for (var y = 0; y < height; y++) {
          blob.data[d * width * height + x * height + y] = data[d][x][y];
        }
      }
    }

    var poolingLayer = new PoolingLayer(width, height, depth, null, windowSize, stride, pad);
    var result = poolingLayer.fprop(blob);

    var expectedDepth = expected.length;
    var expectedWidth = expected[0].length;
    var expectedHeight = expected[0][0].length;

    if (result.depth != expectedDepth) {
      console.log("FAIL - testPooling - " + name + " wrong depth");
    }

    if (result.width != expectedWidth) {
      console.log("FAIL - testPooling - " + name + " wrong width");
    }

    if (result.height != expectedHeight) {
      console.log("FAIL - testPooling - " + name + " wrong height");
    }

    for (var d = 0; d < expectedDepth; d++) {
      for (var x = 0; x < expectedWidth; x++) {
        for (var y = 0; y < expectedHeight; y++) {
          var val = result.data[d * expectedWidth * expectedHeight + x * expectedHeight + y];
          var exp = expected[d][x][y];
          if (val != exp) {
            console.log("FAIL - testPooling fprop - " + name + " wrong value: " + val + ", expected: " + exp);
          }
        }
      }
    }

    var nextBlob = new Blob(result.width, result.height, result.depth);
    var i = 0;
    for (var d = 0; d < expectedDepth; d++) {
      for (var x = 0; x < expectedWidth; x++) {
        for (var y = 0; y < expectedHeight; y++) {
          nextBlob.delta[i++] = i;
        }
      }
    }

    var thisBlob = poolingLayer.bprop(nextBlob);
    for (var d = 0; d < depth; d++) {
      for (var x = 0; x < width; x++) {
        for (var y = 0; y < height; y++) {
          var val = thisBlob.delta[d * width * height + x * height + y];
          var exp = expectedDelta[d][x][y];
          if (val != exp) {
            console.log("FAIL - testPooling bprop - " + name + " wrong value: " + val + ", expected: " + exp);
          }
        }
      }
    }
  }

  test("4x4x1,window=2x2,stride=2,pad=0", 2, 2, 0,
    [
      [
        [ 0, 1, 2, 3, ],
        [ 4, 5, 6, 7, ],
        [ 8, 9, 0, 1, ],
        [ 2, 3, 4, 5, ],
      ]
    ],
    [
      [
        [ 5, 7, ],
        [ 9, 5, ],
      ]
    ],
    [
      [
        [ 0, 0, 0, 0, ],
        [ 0, 1, 0, 2, ],
        [ 0, 3, 0, 0, ],
        [ 0, 0, 0, 4, ],
      ]
    ]
  );

  test("4x4x2,window=2x2,stride=2,pad=0", 2, 2, 0,
    [
      [
        [ 0, 1, 2, 3, ],
        [ 4, 5, 6, 7, ],
        [ 8, 9, 0, 1, ],
        [ 2, 3, 4, 5, ],
      ],
      [
        [ 7, 8, 3, 4, ],
        [ 1, 0, 9, 5, ],
        [ 2, 5, 4, 7, ],
        [ 4, 5, 3, 6, ],
      ],
    ],
    [
      [
        [ 5, 7, ],
        [ 9, 5, ],
      ],
      [
        [ 8, 9, ],
        [ 5, 7, ],
      ],
    ],
    [
      [
        [ 0, 0, 0, 0, ],
        [ 0, 1, 0, 2, ],
        [ 0, 3, 0, 0, ],
        [ 0, 0, 0, 4, ],
      ],
      [
        [ 0, 5, 0, 0, ],
        [ 0, 0, 6, 0, ],
        [ 0, 7, 0, 8, ],
        [ 0, 0, 0, 0, ],
      ],
    ]
  );

  test("5x5x1,window=2x2,stride=3,pad=0", 2, 3, 0,
    [
      [
        [ 0, 1, 2, 3, 4, ],
        [ 5, 6, 7, 8, 9, ],
        [ 9, 8, 7, 6, 5, ],
        [ 4, 3, 2, 1, 0, ],
        [ 0, 1, 2, 3, 4, ]
      ]
    ],
    [
      [
        [ 6, 9, ],
        [ 4, 4, ],
      ]
    ],
    [
      [
        [ 0, 0, 0, 0, 0, ],
        [ 0, 1, 0, 0, 2, ],
        [ 0, 0, 0, 0, 0, ],
        [ 3, 0, 0, 0, 0, ],
        [ 0, 0, 0, 0, 4, ]
      ]
    ]
  );

  test("5x5x1,window=3x3,stride=2,pad=0", 3, 2, 0,
    [
      [
        [ 0, 1, 2, 3, 4, ],
        [ 5, 6, 7, 8, 9, ],
        [ 9, 8, 7, 6, 5, ],
        [ 4, 3, 2, 1, 0, ],
        [ 0, 1, 2, 3, 4, ]
      ]
    ],
    [
      [
        [ 9, 9, ],
        [ 9, 7, ],
      ]
    ],
    [
      [
        [ 0, 0, 0, 0, 0, ],
        [ 0, 0, 0, 0, 2, ],
        [ 4, 0, 4, 0, 0, ],
        [ 0, 0, 0, 0, 0, ],
        [ 0, 0, 0, 0, 0, ]
      ]
    ]
  );

  test("2x2x1,window=2x2,stride=2,pad=1", 2, 2, 1,
    [
      [
        [ 5, 6, ],
        [ 9, 0, ],
      ]
    ],
    [
      [
        [ 5, 6, ],
        [ 9, 0, ],
      ]
    ],
    [
      [
        [ 1, 2, ],
        [ 3, 4, ],
      ]
    ]
  );
})();
