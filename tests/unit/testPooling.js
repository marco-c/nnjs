var assert = require('assert');

import * as nnjs from '../../nnjs';

function doTest(name, windowSize, stride, pad, data, expected, expectedDelta) {
  test(name, function() {
    var depth = data.length;
    var width = data[0].length;
    var height = data[0][0].length;

    var blob = new nnjs.Blob(width, height, depth);
    for (var d = 0; d < depth; d++) {
      for (var x = 0; x < width; x++) {
        for (var y = 0; y < height; y++) {
          blob.data[d * width * height + x * height + y] = data[d][x][y];
        }
      }
    }

    var poolingLayer = new nnjs.PoolingLayer(null, windowSize, stride, pad);
    poolingLayer.init(width, height, depth);
    var result = poolingLayer.fprop(blob);

    var expectedDepth = expected.length;
    var expectedWidth = expected[0].length;
    var expectedHeight = expected[0][0].length;

    assert.equal(result.width, expectedWidth, "wrong width");
    assert.equal(result.height, expectedHeight, "wrong height");
    assert.equal(result.depth, expectedDepth, "wrong depth");

    for (var d = 0; d < expectedDepth; d++) {
      for (var x = 0; x < expectedWidth; x++) {
        for (var y = 0; y < expectedHeight; y++) {
          assert.equal(result.data[d * expectedWidth * expectedHeight + x * expectedHeight + y],
                       expected[d][x][y],
                       "fprop - wrong value");
        }
      }
    }

    var nextBlob = new nnjs.Blob(result.width, result.height, result.depth);
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
          assert.equal(thisBlob.delta[d * width * height + x * height + y],
                       expectedDelta[d][x][y],
                       "bprop - wrong value");
        }
      }
    }
  });
}

suite('PoolingLayer', function() {
  doTest("4x4x1,window=2x2,stride=2,pad=0", 2, 2, 0,
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

  doTest("4x4x2,window=2x2,stride=2,pad=0", 2, 2, 0,
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

  doTest("5x5x1,window=2x2,stride=3,pad=0", 2, 3, 0,
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

  doTest("5x5x1,window=3x3,stride=2,pad=0", 3, 2, 0,
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

  doTest("2x2x1,window=2x2,stride=2,pad=1", 2, 2, 1,
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
});
