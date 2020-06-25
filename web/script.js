// const labels = [
//     'accordion',     'airplanes',      'anchor',        'ant',
//     'barrel',        'bass',           'beaver',        'binocular',
//     'bonsai',        'brain',          'brontosaurus',  'buddha',
//     'butterfly',     'camera',         'cannon',        'car_side',
//     'ceiling_fan',   'cellphone',      'chair',         'chandelier',
//     'cougar_body',   'cougar_face',    'crab',          'crayfish',
//     'crocodile',     'crocodile_head', 'cup',           'dalmatian',
//     'dollar_bill',   'dolphin',        'dragonfly',     'electric_guitar',
//     'elephant',      'emu',            'euphonium',     'ewer',
//     'Faces',         'Faces_easy',     'ferry',         'flamingo',
//     'flamingo_head', 'garfield',       'gerenuk',       'gramophone',
//     'grand_piano',   'hawksbill',      'headphone',     'hedgehog',
//     'helicopter',    'ibis',           'inline_skate',  'joshua_tree',
//     'kangaroo',      'ketch',          'lamp',          'laptop',
//     'Leopards',      'llama',          'lobster',       'lotus',
//     'mandolin',      'mayfly',         'menorah',       'metronome',
//     'minaret',       'Motorbikes',     'nautilus',      'octopus',
//     'okapi',         'pagoda',         'panda',         'pigeon',
//     'pizza',         'platypus',       'pyramid',       'revolver',
//     'rhino',         'rooster',        'saxophone',     'schooner',
//     'scissors',      'scorpion',       'sea_horse',     'snoopy',
//     'soccer_ball',   'stapler',        'starfish',      'stegosaurus',
//     'stop_sign',     'strawberry',     'sunflower',     'tick',
//     'trilobite',     'umbrella',       'watch',         'water_lilly',
//     'wheelchair',    'wild_cat',       'windsor_chair', 'wrench',
//     'yin_yang'
//   ];

const labels = ['Airplanes','Bonsai','Car side','Leopard','Motorbike'];

let model;

(async () => {
    model = await tf.loadLayersModel('../train/model/model.json');
    console.log('Loaded');
})();

function UploadImage() {
    let reader = new FileReader();

    reader.onload = (() => {
        let dataURL = reader.result;
        document.querySelector('#selected-image').src = dataURL;
    });

    let file = document.querySelector('#in-image').files[0];
    reader.readAsDataURL(file);
}

async function Predict() {
    let image = document.querySelector('#selected-image');
    let tensor = tf.browser.fromPixels(image, 3)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

    let predictions = await model.predict(tensor).data();

    let top10 = Array.from(predictions)
        .map((p, i) => [labels[i], p])
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(item => `<tr><td class="text-capitalize">${item[0].replace(/_/g, ' ')}</td><td class="text-right">${(item[1] * 100).toFixed(2)}%</td></tr>`)
        .join('');

    document.querySelector('#prediction-list').innerHTML = top10;
}