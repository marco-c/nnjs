/* -*- Mode: Java; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set shiftwidth=2 tabstop=2 autoindent cindent expandtab: */

'use strict';

var Rand = (function() {
  /*
    Method from:
      Knuth Sec. 3.4.1 p. 117
      Marsaglia and Bray, ``A Convenient Method for Generating Normal Variables''
  */
  var cachedVal = null;
  function gaussRand() {
    if (cachedVal) {
      var ret = cachedVal;
      cachedVal = null;
      return ret;
    }

    var s;
  	do {
  		var v1 = 2 * Math.random() - 1;
  		var v2 = 2 * Math.random() - 1;
  		s = v1 * v1 + v2 * v2;
  	} while (s >= 1 || s == 0);

    cachedVal = v2 * Math.sqrt(-2 * Math.log(s) / s);
  	return v1 * Math.sqrt(-2 * Math.log(s) / s);
  }

  function randn(mean, std) {
    return mean + gaussRand() * std;
  }

  return {
    randn: randn,
  }
})();
