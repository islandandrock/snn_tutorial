from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt('keras_tutorial/pima-indians-diabetes.csv', delimiter=',')
# Split into input, output
X = dataset[:,0:8]
y = dataset[:,8]

# Define model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit model (train)
model.fit(X, y, epochs=150, batch_size=10)
# Test on same training data
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))