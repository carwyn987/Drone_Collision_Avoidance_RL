/**
 * Sigmoid function
 * @param {Number} z 
 */
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z/4));
}

/**
 * Sums all the values including and after an index in an array
 * @param {Array} arr An array
 * @param {Number} index The index
 */
function sumRest(arr, index) {
    let s = 0;
    for(let i = index; i<arr.length; i++){
        s+= arr[i];
    }
    return s;
}

export default class Model {
    /**
     * Define a new model with specified parameters.
     * @param {Number} numHiddenLayerElements Number of nodes in the hidden layer
     * @param {Number} numStateVars Number of state variables which define environemnt
     * @param {Number} numActions Number of actions the drone can take
     * @param {Number} batchSize Number of samples of memory to train model on after each game
     */
    constructor(numHiddenLayerElements, numStateVars, numActions, batchSize, discountRate) {
        this.hiddenLayers = numHiddenLayerElements;
        this.numStateVars = numStateVars;
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.discountRate = discountRate;
        this.learningRate = 0.001;

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

        this.network.add(tf.layers.dense({
            units: this.hiddenLayers,
            activation: 'relu',
        }))

        //Add a densely connected output layer
        this.network.add(tf.layers.dense({
            units: this.numActions
        }));

        // Print summary and compile network
        this.network.summary();

        const optimizer = tf.train.adam(this.learningRate);
        this.network.compile({optimizer: optimizer, loss: 'meanSquaredError'});
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
     * @param {Number} eps Decimal representing chance of randomly choosing action
     * @returns {Number} The action chosen by the model
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

    /**
     * Train model on data saved in memory
     * @param {Object} memory Memory object
     * @param {Number} numFrames Number of frames of last game
     */
    async commenceTraining(memory, sum){
        let batch = memory.sample(this.batchSize);
        // filter out states from batch
        let states = batch.map(([state, , ]) => state);
        // Actions at each state
        let actions = batch.map(([ ,action, ]) => action);

        let rewards = batch.map(([ , , reward]) => reward);

        // Predict the values of each action at each state
        let qsa = states.map((state) => this.predict(state));

        let x = new Array();
        let y = new Array(); 

        // Update the states rewards with the discounted next states rewards
        batch.forEach(
            ([state, action, reward], index) => {
                let currentQ = qsa[index];
                currentQ = currentQ.dataSync();
                
                if(rewards[index+5] && index+5 < memory.maxMemory){
                    currentQ[action] = sigmoid(sumRest(rewards, index));
                    
                    console.log(sumRest(rewards, index),currentQ[action])
                    x.push(state.dataSync());
                    y.push(currentQ);
                }
            }
        );

        // Reshape the batches to be fed to the network
        x = tf.tensor2d(x, [x.length, this.numStateVars])
        y = tf.tensor2d(y, [y.length, this.numActions])

        // Learn the Q(s, a) values given associated discounted rewards
        await this.train(x, y);
    }
}