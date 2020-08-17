# Lesson 2
Building a simple model for predicting Fahrenheit given the input in Celcius

```
l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
```

- `input_shape=[1]` — This specifies that the input to this layer is a single value.
- `units=1` — This specifies the number of neurons in the layer.
- `loss='mean_squared_error` — measuring how far from prediction
- `optimizer=tf.keras.optimizers.Adam(0.1))` — adjusting internal values in order to reduce the loss
        - `0.1` is the learning rate

Displaying the training statistics
```
import matplotlib.pyplot as plt
ply.xlabel('Epoch Number')
ply.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
``` 

Predicting the values
`print(model.predict([100.0]))`

Getting the weights
`l0.get_weights()))`

# Lesson 3
Converting a 2d Image into 1d vector
Input
`tf.keras.layers.Flatten(input_shape=(28, 28, 1))`

Hidden Layer
`tf.keras.layers.Dense(128, actication=tf.nn.relu)`

Output with softmax
`tf.keras.layers.Dense(10, activation=tf.nn.softmax)`


The Rectified Linear Unit(relu) - gives the model more power making it non-linear

### Classifying images of clothing (MNIST)
Loading data directly from TensorFlow using the Dataset API
```
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
```

Preproccessing the data
Normalize the range of pixel which is in the range of `[0,255]`
```
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()
```

Plotting the image to see what it's like
```
# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
```

Building the model
```
model = tf.keras.Sequential([
    tf.keras.Flatten(input_shape=(28,28,1)),
    tf.keras.Dense(128, activation=tf.nn.relu),
    tf.keras.Dense(10, activation=tf.nn.softmax)
])
```
Compile the model
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
```
Train the model
```
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
```
1. `dataset.repear()` - repeat forever
2. `dataset.shuffle(60000)` - randomizes the order
3. `dataset.batch(32)` - tells `model.fit` to use batches of 32 images when updating the model variables.

`model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))`
Training is performed by calling the `model.fit()` method
1. Feed the training data to model using `train_dataset`
2. The model learns to associate images and labels
3. `epochs=5` 5 full iterations of the training dataset

Evaluate accuracy
```
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)
```

Making predictions and exploration
```
for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)
```
Graph this to look at the full set of 10 class predictions
```
def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  ```

Looking at the 0th image, preidctions and prediction array
```
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
```

Plotting several images
```
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
```