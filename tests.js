/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

function test(name) {
  console.log(name);
  load(name + '.js');
}

test('testPooling');
test('testConvolution');
test('testNOT');
test('testNOTWithReLU');
test('testNOTTwoLayers');
test('testOR');
test('testORTwoLayers');
test('testAND');
test('testXOR');
test('testXORWithMomentum');
test('testVectorNOT');
test('testIRIS');
test('testMNIST');
