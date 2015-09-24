import cPickle, gzip, numpy

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_images = train_set[0]
train_set_labels = train_set[1]

def generateImagesVar(imageSet):
  images = []
  for i in range(0,len(imageSet)):
    image = []
    for j in range(0,len(imageSet[i])):
      image.append(str(imageSet[i][j]))

    images.append("[" + ",".join(image) + "]")

  return "[" + ",".join(images) + "]"

def generateLabelsVar(labelSet):
  labels = []

  for i in range(0,len(labelSet)):
    labels.append(str(labelSet[i]))

  return "[" + ",".join(labels) + "]"

open('mnist_train_images.json', 'w').write(generateImagesVar(train_set[0]))
open('mnist_test_images.json', 'w').write(generateImagesVar(test_set[0]))
open('mnist_train_labels.json', 'w').write(generateLabelsVar(train_set[1]))
open('mnist_test_labels.json', 'w').write(generateLabelsVar(test_set[1]))

print(len(train_set))
print(len(train_set[0]))
print(len(train_set[1]))
print(train_set[1][0])
print(len(train_set[0][0]))
