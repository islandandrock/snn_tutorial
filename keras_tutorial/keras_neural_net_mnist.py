from numpy import loadtxt
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt('homemade_snn/mnist_dataset/mnist_train.csv', delimiter=',')
# Split into input, output
X = (dataset[:,1:785] / 255.0 * 0.99) + 0.01
y = dataset[:,0]
ylist = []

for i in range(len(y)):    
    y2 = numpy.zeros(10) + 0.01
    y2[int(y[i])] = 0.99
    ylist.append(y2)

ylist = numpy.stack(ylist)
print(type(X), type(numpy.stack(ylist)))

# Define model
model = Sequential()
model.add(Dense(100, input_shape=(784,), activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model (train)
model.fit(X, ylist, epochs=15, batch_size=10)
# Test on same training data

dataset = loadtxt('homemade_snn/mnist_dataset/mnist_test.csv', delimiter=',')
# Split into input, output
X = (dataset[:,1:785] / 255.0 * 0.99) + 0.01
y = dataset[:,0]
ylist = []

for i in range(len(y)):    
    y2 = numpy.zeros(10) + 0.01
    y2[int(y[i])] = 0.99
    ylist.append(y2)

ylist = numpy.stack(ylist)

_, accuracy = model.evaluate(X, ylist)
print('Accuracy: %.2f' % (accuracy*100))
