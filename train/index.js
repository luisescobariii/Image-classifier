const dataset = require('./dataset');
const model = require('./model');

async function Run(epochs, batchSize) {
    dataset.Load();

    let data = dataset.GetTrainData();
    console.log('Training images shape: ' + data.images.shape);
    console.log('Training labels shape: ' + data.labels.shape);
    
    model.summary();

    const validationSplit = 0.2;
    await model.fit(data.images, data.labels, {
        epochs,
        batchSize,
        validationSplit
    });

    data = dataset.GetTestData();
    const evalOutput = model.evaluate(data.images, data.labels);
    console.log('Evaluation result');
    console.log('Loss = ' + evalOutput[0].dataSync()[0].toFixed(3));
    console.log('Accuracy = ' + evalOutput[1].dataSync()[0].toFixed(3));

    await model.save('file://./model');
    console.log('Model saved');
}

Run(2, 32);