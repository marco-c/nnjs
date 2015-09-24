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

export default {
  argmax,
  shuffle,
};
