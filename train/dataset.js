'use extrict';

const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

const dataDir = './data';

const maxCategories = 5;

function LoadImages(testSize = 0.2) {
    const data = {
        train: { images: [], labels: [] },
        test: { images: [], labels: [] },
    };

    let maxFolders = maxCategories * 1;
    
    const directories = fs.readdirSync(dataDir);
    let directoryIndex = -1;
    const images = [];
    const labels = [];
    const testFiles = [];
    for (const dir of directories) {
        directoryIndex++;
        if (maxFolders-- === 0) { break; }
        const files = fs.readdirSync(path.join(dataDir, dir));
        let testFileCount = Math.floor(files.length * testSize);
        
        for (const file of files) {
            images.push(path.join(dataDir, dir, file));
            labels.push(directoryIndex);
            testFiles.push(testFileCount-- > 0);
        }
    }
    
    const totalFiles = images.length;
    for (let i = 0; i < totalFiles; i++) {
        const percent = ((i / totalFiles) * 100).toFixed(0);
        process.stdout.cursorTo(0);
        process.stdout.write(`Loading images... ${i}/${totalFiles} ${percent}%`);
        
        const buffer = fs.readFileSync(images[i]);
        const tensor = tf.node.decodeJpeg(buffer, 3)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
        
        if (testFiles[i]) {
            data.test.images.push(tensor);
            data.test.labels.push(labels[i]);
        } else {
            data.train.images.push(tensor);
            data.train.labels.push(labels[i]);
        }
    }
    
    return data;
}

class Dataset {

    constructor() {
        this.testData = [];
        this.trainData = [];
    }

    Load() {
        console.clear();
        const data = LoadImages();
        this.testData = data.test;
        this.trainData = data.train;
        console.log();
    }

    GetTrainData() {
        return {
            images: tf.concat(this.trainData.images),
            labels: tf.oneHot(tf.tensor1d(this.trainData.labels, 'int32'), maxCategories).toFloat(),
        };
    }

    GetTestData() {
        return {
            images: tf.concat(this.testData.images),
            labels: tf.oneHot(tf.tensor1d(this.testData.labels, 'int32'), maxCategories).toFloat(),
        };
    }

}

module.exports = new Dataset();