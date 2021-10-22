export default class Model {
    /**
     * Define a new model with specified parameters.
     * @param {Integer} numHiddenLayers Number of nodes in the first and only hidden layer
     * @param {Integer} numStates Number of states which the drone and ball can be in
     * @param {Integer} numActions Number of actions the drone can take
     * @param {Integer} batchSize Number of samples of memory to train model on after each game
     */
    constructor(numHiddenLayers, numStates, numActions, batchSize) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.batchSize = batchSize;
        this.hiddenLayers = [numHiddenLayers];
        this.discountRate = .9;
        this.numDiscounts = 10;

        // Calculate Discount Scalar
        let sum = 100;
        for(let i = 1; i<this.numDiscounts; i++){
            sum += (this.discountRate**i)*100;
        }

        this.discountScalar = 100/(sum);

        this.defineModel();
    }

    /**
     * Define a new model in the constructed Model.
     * Defines this.network
     */
    defineModel(){
        // Define Network
        this.network = tf.sequential();

        // For the number of hidden layers defined in constructer, add a hidden layer with the set size.
        this.hiddenLayers.forEach((hiddenLayerSize, i) => {
            console.log("numStates: ", this.numStates)
            console.log("i: ", i)
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

    /**
     * Predict the passed in state and return the prediction (action)
     * @param {Object Array} states State or array of states
     * @return Action
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
                // let ret = this.network.predict(state).argMax(1).dataSync()[0];
                // return ret;
                let logits = this.network.predict(state);
                let sigmoid = tf.sigmoid(logits);
                let probs = tf.div(sigmoid, tf.sum(sigmoid));
                return tf.multinomial(probs, 1).dataSync()[0];
            });
        }
    }

    /**
     * Pull samples from memory, and train network on these samples.
     * @param {Object} memory Memory object to pull samples from
     */
    async processAndTrain(memory, frames) {
        let batch = memory.sample(this.batchSize);
        // filter out states from batch
        let states = batch.map(([state, , , ]) => state);
        // filter out nextStates from batch
        let nextStates = batch.map(([ , , , nextState]) => nextState ? nextState : tf.zeros([this.model.numStates]));
        // Actions at each state
        let actions = batch.map(([ , action, , ]) => action);

        // Predict the values of each action at each state
        let qsa = states.map((state) => this.predict(state));
        // Predict the values of each action at each next state
        let qsad = nextStates.map((nextState) => this.predict(nextState));

        

        let x = new Array();
        let y = new Array(); 

        // Update the states rewards with the discounted next states rewards
        batch.forEach(
            ([state, action, reward, nextState], index) => {
                let currentQ = qsa[index];
                currentQ = currentQ.dataSync();
                
                // currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward; //reward + this.discountRate * qsad[index].max().dataSync() // + this.discountRate * qsad[index].dataSync()[action]
                // x.push(state.dataSync());
                // y.push(currentQ);
                
                if(batch[index-this.numDiscounts]){

                    let origReward = currentQ[action];

                    let sum = reward;
                    for(let i = 1; i<=this.numDiscounts; i++){
                        sum += (this.discountRate**i)*qsa[index-i].dataSync()[actions[index - i]]
                    }
                    currentQ[action] = currentQ[action] + 0.003*this.discountScalar*sum;

                    x.push(state.dataSync());
                    y.push(currentQ);
                    console.log("Reward: ", reward, "Orig Choice: ", origReward, "New Choice: ", currentQ[action], "Choice Change: ", (currentQ[action] -origReward), ", Action: ", action);
                }
            }
        );

        // Reshape the batches to be fed to the network
        x = tf.tensor2d(x, [x.length, this.numStates])
        y = tf.tensor2d(y, [y.length, this.numActions])

        // Learn the Q(s, a) values given associated discounted rewards
        await this.train(x, y);
    }
}