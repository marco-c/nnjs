var assert = require('assert');

import * as nnjs from '../../nnjs';

function doTest(name, windowSize, stride, pad, weights, biases, data, expected, expectedDelta) {
  test(name, function() {
    var outputDepth = weights.length;
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

    var convolutionLayer = new nnjs.ConvolutionLayer(outputDepth, windowSize, stride, pad);
    convolutionLayer.init(width, height, depth);

    for (var num = 0; num < outputDepth; num++) {
      for (var d = 0; d < depth; d++) {
        for (var x = 0; x < windowSize; x++) {
          for (var y = 0; y < windowSize; y++) {
            var idx = num * depth * windowSize * windowSize +
                      d * windowSize * windowSize +
                      x * windowSize +
                      y;
            convolutionLayer.weights.data[idx] = weights[num][d][x][y];
          }
        }
      }
    }

    for (var i = 0; i < outputDepth; i++) {
      convolutionLayer.biases.data[i] = biases[i];
    }

    var result = convolutionLayer.fprop(blob);

    var expectedDepth = expected.length;
    var expectedWidth = expected[0].length;
    var expectedHeight = expected[0][0].length;

    assert.equal(result.width, expectedWidth, "fprop - wrong width");
    assert.equal(result.height, expectedHeight, "fprop - wrong height");
    assert.equal(result.depth, expectedDepth, "fprop - wrong depth");

    for (var d = 0; d < expectedDepth; d++) {
      for (var x = 0; x < expectedWidth; x++) {
        for (var y = 0; y < expectedHeight; y++) {
          assert.equal(result.data[d * expectedWidth * expectedHeight + x * expectedHeight + y],
                       expected[d][x][y],
                       "frop - wrong value");
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

    var thisBlob = convolutionLayer.bprop(nextBlob);

    var expectedDeltaDepth = expectedDelta.length;
    var expectedDeltaWidth = expectedDelta[0].length;
    var expectedDeltaHeight = expectedDelta[0][0].length;

    assert.equal(thisBlob.width, expectedDeltaWidth, "bprop - wrong width");
    assert.equal(thisBlob.height, expectedDeltaHeight, "bprop - wrong height");
    assert.equal(thisBlob.depth, expectedDeltaDepth, "bprop - wrong depth");

    for (var d = 0; d < depth; d++) {
      for (var x = 0; x < width; x++) {
        for (var y = 0; y < height; y++) {
          assert.equal(thisBlob.delta[d * width * height + x * height + y],
                       expectedDelta[d][x][y],
                       "brop - wrong delta");
        }
      }
    }
  });
}

suite('ConvolutionLayer', function() {
  // TODO: Test bprop also for weights and biases

  doTest("4x4x1,window=2x2,filters=1,stride=2,pad=0", 2, 2, 0,
    [
      [
        [
          [ 1, 2, ],
          [ 3, 4, ],
        ]
      ]
    ],
    [ 99 ],
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
        [ 133, 153, ],
        [ 143, 133, ],
      ]
    ],
    [
      [
        [ 1, 2, 2, 4, ],
        [ 3, 4, 6, 8, ],
        [ 3, 6, 4, 8, ],
        [ 9, 12, 12, 16, ],
      ]
    ]
  );

  doTest("4x4x2,window=2x2,filters=1,stride=2,pad=0", 2, 2, 0,
    [
      [
        [
          [ 1, 2, ],
          [ 3, 4, ],
        ],
        [
          [ 5, 6, ],
          [ 7, 8, ],
        ],
      ]
    ],
    [ 99 ],
    [
      [
        [ 0, 1, 2, 3, ],
        [ 4, 5, 6, 7, ],
        [ 8, 9, 0, 1, ],
        [ 2, 3, 4, 5, ],
      ],
      [
        [ 9, 8, 7, 6, ],
        [ 5, 4, 3, 2, ],
        [ 1, 0, 0, 1, ],
        [ 2, 3, 4, 5, ],
      ]
    ],
    [
      [
        [ 293, 261, ],
        [ 186, 207, ],
      ],
    ],
    [
      [
        [ 1, 2, 2, 4, ],
        [ 3, 4, 6, 8, ],
        [ 3, 6, 4, 8, ],
        [ 9, 12, 12, 16, ],
      ],
      [
        [ 5, 6, 10, 12, ],
        [ 7, 8, 14, 16, ],
        [ 15, 18, 20, 24, ],
        [ 21, 24, 28, 32, ],
      ]
    ]
  );

  doTest("4x4x1,window=2x2,filters=2,stride=2,pad=0", 2, 2, 0,
    [
      [
        [
          [ 1, 2, ],
          [ 3, 4, ],
        ]
      ],
      [
        [
          [ 5, 6, ],
          [ 7, 8, ],
        ]
      ],
    ],
    [ 99, 15, ],
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
        [ 133, 153, ],
        [ 143, 133, ],
      ],
      [
        [ 89, 141, ],
        [ 147, 89, ],
      ],
    ],
    [
      [
        [ 1+25, 2+30, 2+30, 4+36, ],
        [ 3+35, 4+40, 6+42, 8+48, ],
        [ 3+35, 6+42, 4+40, 8+48, ],
        [ 9+49, 12+56, 12+56, 16+64, ],
      ],
    ]
  );

  doTest("6x6x3,window=3x3,filters=3,stride=2,pad=1", 3, 2, 1,
    [
      [
        [
          [ 1, 1, 1, ],
          [ 1, -1, 1, ],
          [ -1, 0, -1, ],
        ],
        [
          [ 1, 1, 1, ],
          [ 0, 0, 0, ],
          [ -1, -1, -1, ],
        ],
        [
          [ 1, 0, 0, ],
          [ 0, 0, -1, ],
          [ 1, 1, 1, ],
        ],
      ],
      [
        [
          [ -1, 1, -1, ],
          [ 0, 1, 1, ],
          [ 1, 1, 1, ],
        ],
        [
          [ -1, 1, 1, ],
          [ 0, 1, 0, ],
          [ 0, -1, 0, ],
        ],
        [
          [ 1, -1, 0, ],
          [ 0, -1, 0, ],
          [ 1, 1, -1, ],
        ],
      ],
    ],
    [ 1, 0, ],
    [
      [
        [ 2, 1, 1, 1, 0, ],
        [ 0, 1, 0, 0, 2, ],
        [ 1, 1, 1, 2, 0, ],
        [ 0, 1, 0, 2, 2, ],
        [ 1, 1, 1, 2, 0, ],
      ],
      [
        [ 2, 0, 2, 0, 0, ],
        [ 0, 2, 2, 0, 1, ],
        [ 0, 0, 2, 2, 1, ],
        [ 1, 0, 0, 1, 2, ],
        [ 0, 0, 0, 2, 2, ],
      ],
      [
        [ 0, 2, 0, 2, 2, ],
        [ 1, 0, 2, 1, 0, ],
        [ 2, 2, 0, 2, 0, ],
        [ 1, 1, 0, 1, 1, ],
        [ 1, 0, 1, 2, 0, ],
      ],
    ],
    [
      [
        [ -4, -2, 2, ],
        [ 2, 4, 4, ],
        [ 3, 6, 11, ],
      ],
      [
        [ 7, 4, 0, ],
        [ 0, 5, 9, ],
        [ 0, 1, 3, ],
      ],
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
});
