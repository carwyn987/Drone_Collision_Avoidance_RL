export default class Model {
    /**
     * Define a new model with specified parameters.
     * @param {Integer} numHiddenLayerElements Number of nodes in the hidden layer
     * @param {Integer} numStateVars Number of state variables which define environemnt
     * @param {Integer} numActions Number of actions the drone can take
     * @param {Integer} batchSize Number of samples of memory to train model on after each game
     */
    constructor(numHiddenLayerElements, numStateVars, numActions, batchSize) {
        this.hiddenLayers = numHiddenLayerElements;
        this.numStateVars = numStateVars;
        this.numActions = numActions;
        this.batchSize = batchSize;

        this.defineModel();
    }

    /**
     * Define a new model in the constructed Model.
     * Defines this.network and specifies the network parameters.
     */
    defineModel(){
        // Define Network
        this.network = tf.sequential();

        // Add a hidden layer with the set size (units), and input size (inputShape)
        this.network.add(tf.layers.dense({
            units: this.hiddenLayers,
            activation: 'relu',
            inputShape: this.numStateVars
        }))

        //Add a densely connected output layer
        this.network.add(tf.layers.dense({
            units: this.numActions
        }));

        // Print summary and compile network
        this.network.summary();
        this.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
    }

    /**
     * Predict actions given an array of states 
     * @param {Object} states Array of states
     * @return {Tensor} Return the predictions (actions)
     */
    predict(states) {
        return tf.tidy(() => this.network.predict(states));
    }

    /**
     * Fit batches upon the network and train
     * @param {Tensor2D} xBatch States
     * @param {Tensor2D} yBatch Actions
     */
    async train(xBatch, yBatch) {
        await this.network.fit(xBatch, yBatch);
    }

    /**
     * Choose an action either by random or by prediction (choose from eps)
     * We want to begin by exploring a lot, and then as we become more confident we will set into a strategy.
     * @param {Object} state Current state to predict action from
     * @param {Float} eps Decimal representing chance of randomly choosing action
     * @returns {number} The action chosen by the model
     */
    chooseAction(state, eps) {
        if(Math.random() < eps){
            // Choose a random action
            return Math.floor(Math.random() * this.numActions);
        }else {
            // tf.tidy disposes of tensors created during exection (no garbage collection)
            return tf.tidy(() => {
                return this.network.predict(state).argMax(1).dataSync()[0];
            });
        }
    }
}