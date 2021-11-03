import Drone from './drone.js';
import Ball from './ball.js';
import Memory from './memory.js';
import Model from './model.js';
import draw from './draw.js';
import sleep from './sleep.js';
import calculateReward from './reward.js';
import rewardRange from './visual_extras.js';

// Set up environment/global variables
let MEMORY_SIZE = 500;
let GRAVITY = 0.02;
let NUM_SIMULATIONS = 99999;
let RAND_ACTION_PROB = 0.99;
let DISCOUNT_RATE = 0.9;
let MAX_FRAMES = 1000;

/**
 * Begins execution of main program loop in async function.
 */
async function beginExecution(){
    // Set up canvas (width, height, etc).
    let canvas = document.querySelector("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    let ctx = canvas.getContext('2d');

    // Create a memory object for storing drone state
    let memory = new Memory(MEMORY_SIZE);

    // Create a new model object for predicting drone action
    let model = new Model(50, 6, 2, 100, DISCOUNT_RATE); // Currently set to 10 hidden layer nodes, 2 states (drone y, vy), 2 actions (up, down), 100 batch size
    
    // Set up variable for number of iterations of training
    let sims = 0;

    // Attempt to load a model saved in local storage
    try{
        let network = await tf.loadLayersModel('localstorage://my-model-2');
        model.network = network;
        model.network.summary();
        model.network.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        sims = localStorage.getItem('numIterations2');
        RAND_ACTION_PROB = 0;
    }catch(err){
        console.log("No model exists, generating model with random parameters.");
    }

    // Create a drone and render its current position
    let drone = new Drone(canvas, ctx);
    drone.renderDrone(ctx);

    // Create a ball and render it
    let ball = new Ball(20,300,2.5,-1.9,30);
    ball.renderBall(ctx);

    // Define the center of the canvas
    let center = {
        x: canvas.width/2,
        y: canvas.height/2
    }

    // Allocate droneState variable in memory
    let droneState;
    let totalState;
    let action;
    let reward;
    let numFrames;
    let sum;

    // Run NUM_SIMULATIONS simulations
    for( ;sims<NUM_SIMULATIONS; sims++){
        console.log(sims)
        // Run the current simulation until drone crashes
        numFrames = 0;
        sum = 0;
        let crashed = false;
        while(!crashed && numFrames < MAX_FRAMES){
            // Saves browser from crashing
            await sleep(0);

            // Get current drone state
            droneState = drone.getState(canvas);
            totalState = tf.concat([droneState, ball.getState(canvas)],1);

            // Choose and perform action
            action = model.chooseAction(totalState, RAND_ACTION_PROB);
            drone.move(action)

            // Get the current calculated reward
            reward = calculateReward(drone, ball, center, canvas.height);

            // Draw on canvas updated parameters
            crashed = draw(canvas, ctx, drone, ball, GRAVITY);

            if(crashed)
                reward = -1;

            console.log(reward)
            // Add to total sum of rewards
            sum += reward;

            // Push the current drone state, action, and reward to memory
            memory.addSample([totalState, action, reward]);

            // Set up green reward range visual
            rewardRange(ctx, center);

            // Increment numFrames
            numFrames++;
        }

        // Reset drone to initial position
        drone.setToMiddle();

        // Reset ball to initial position
        ball.resetBall(canvas.height);

        // Decrement RAND_ACTION_PROB exponentially
        RAND_ACTION_PROB *= 0.99;
        // RAND_ACTION_PROB < 0.1 ? RAND_ACTION_PROB = 0.1 :

        // Commence model training
        model.batchSize = numFrames<MAX_FRAMES?numFrames : MAX_FRAMES;
        model.batchSize = numFrames<MEMORY_SIZE?numFrames : MEMORY_SIZE;
        model.commenceTraining(memory);

        // Save the current model to local storage
        if(sims%500 == 0 && sims>0){
            let saveResult = await model.network.save('localstorage://my-model-2');
            localStorage.setItem('numIterations2', sims);
            console.log("Saved model, iteration: ", sims);
        }

    }
}

/**
 * Execute main program execution on page load
 */
window.onload = function(){
    beginExecution();
}