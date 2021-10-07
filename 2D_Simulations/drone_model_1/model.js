export default class Model {
    constructor(numHiddenLayers, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.hiddenLayers = [numHiddenLayers];

        this.defineModel();
    }

    defineModel(){

        // Define Network
        this.network = tf.sequential();

        // For the number of hidden layers defined in constructer, add a hidden layer with the set size.
        this.hiddenLayers.forEach((hiddenLayerSize, i) => {
            this.network.add(tf.layers.dense({
                units: hiddenLayerSize,
                activation: 'relu',
                inputShape: i === 0 ? [this.numStates] : undefined
            }))
        })

        //Add a densely connected layer
        this.network.add(tf.layers.dense({units: this.numActions}));

        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }

    predict(states) {
        return tf.tidy(() => this.network.predict(states));
    }

    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }

    chooseAction(state, eps) {
        if(Math.random() < eps){
            // Choose a random action
            return Math.floor(Math.random() * this.numActions);
        }else {
            // tf.tidy disposes of tensors created during exection (no garbage collection)
            return tf.tidy(() => {
                return this.network.predict(state).argMax(1).dataSync()[0] - 1;
            });
        }
    }
}