'use extrict';

const tf = require('@tensorflow/tfjs');

const pool = {poolSize: [2, 2]};

const filter1 = {filters: 32, kernelSize: [3, 3], activation: 'relu'};
const filter2 = {filters: 64, kernelSize: [3, 3], activation: 'relu'};
const filter3 = {filters: 128, kernelSize: [3, 3], activation: 'relu'};

const dropoutConv = {rate: 0.3};

const model = tf.sequential();

model.add(tf.layers.conv2d({inputShape: [128, 128, 3], ...filter1}));
model.add(tf.layers.conv2d(filter1));
model.add(tf.layers.maxPooling2d(pool));
model.add(tf.layers.dropout(dropoutConv));

model.add(tf.layers.conv2d(filter2));
model.add(tf.layers.conv2d(filter2));
model.add(tf.layers.conv2d(filter2));
model.add(tf.layers.maxPooling2d(pool));
model.add(tf.layers.dropout(dropoutConv));

model.add(tf.layers.conv2d(filter3));
model.add(tf.layers.conv2d(filter3));
model.add(tf.layers.conv2d(filter3));
model.add(tf.layers.maxPooling2d(pool));
model.add(tf.layers.dropout(dropoutConv));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({units: 256, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.3}));
model.add(tf.layers.dense({units: 5, activation: 'softmax'}));

model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
});

module.exports = model;