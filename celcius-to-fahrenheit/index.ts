import * as tf from '@tensorflow/tfjs';

require('@tensorflow/tfjs-node');

async function predict() {
    const celsius = tf.tensor([-40, -10, 0, 8, 15, 22, 38]);
    const fahrenheit = tf.tensor([-40, 14, 32, 46, 59, 72, 100]);

    const network = tf.layers.dense({
        units: 1,
        inputShape: [1]
    });

    const model = tf.sequential({
        layers: [network]
    });

    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.OptimizerConstructors.adam(0.1)
    });

    await model.fit(celsius, fahrenheit, {epochs: 500, verbose: 0});
    console.log('The model is brutal güet trainiert');

    return model.predict(tf.tensor([100]));
}

predict().then((v: any) => {
    console.log('Predicting new fahrenheit');
    v.print()
});


