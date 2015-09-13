/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

var Util = (function() {
  function argmax(array) {
    return array.findIndex((elem) => array.every(other => other <= elem));
  }

  function shuffle(array) {
    for (var i = array.length - 1; i >= 1; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = array[j];
      array[j] = array[i];
      array[i] = tmp;
    }
  }

  return {
    argmax: argmax,
    shuffle: shuffle,
  };
})();
