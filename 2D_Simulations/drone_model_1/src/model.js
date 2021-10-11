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

    // We want to begin by exploring a lot, and then as we become more confident we will set into a strategy.
    chooseAction(state, eps) {
        if(Math.random() < eps){
            // Choose a random action
            return Math.floor(Math.random() * this.numActions);
        }else {
            // tf.tidy disposes of tensors created during exection (no garbage collection)
            return tf.tidy(() => {
                // state.print();
                this.network.predict(state).print();
                // this.network.predict(state).argMax(1).print();
                let ret = this.network.predict(state).argMax(1).dataSync()[0];
                console.log(ret);
                return ret;
            });
        }
    }

    async processAndTrain(memory) {
        let batch = memory.sample(this.batchSize);
        // filter out states from batch
        let states = batch.map(([state, , , ]) => state);
        // filter out nextStates from batch
        let nextStates = batch.map(([state , , , nextState]) => nextState ? nextState : state);

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
                currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max().dataSync() : reward;
                x.push(state.dataSync());
                y.push(currentQ.dataSync());
            }
        );

        // Reshape the batches to be fed to the network
        x = tf.tensor2d(x, [x.length, this.numStates])
        y = tf.tensor2d(y, [y.length, this.numActions])

        // Learn the Q(s, a) values given associated discounted rewards
        await this.train(x, y);

        // console.log(batch);
    }
}