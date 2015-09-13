import cPickle, gzip, numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_images = train_set[0]
train_set_labels = train_set[1]

def generateImagesVar(varName, imageSet):
  gen = "var " + varName + " = ["

  for i in range(0,len(imageSet)):
    gen += "["
    for j in range(0,len(imageSet[i])):
      gen += str(imageSet[i][j]) + ","
    gen += "],"

  gen += "];"

  return gen

def generateLabelsVar(varName, labelSet):
  gen = "var " + varName + " = ["

  for i in range(0,len(labelSet)):
    gen += str(labelSet[i]) + ","

  gen += "];"

  return gen

open('mnist_train_images.js', 'w').write(generateImagesVar("trainImages", train_set[0]))
open('mnist_test_images.js', 'w').write(generateImagesVar("testImages", test_set[0]))
open('mnist_train_labels.js', 'w').write(generateLabelsVar("trainLabels", train_set[1]))
open('mnist_test_labels.js', 'w').write(generateLabelsVar("testLabels", test_set[1]))

print(len(train_set))
print(len(train_set[0]))
print(len(train_set[1]))
print(train_set[1][0])
print(len(train_set[0][0]))
