import Drone from './drone.js';
import Memory from './memory.js';
import draw from './draw.js';
import sleep from './sleep.js';
import Model from './model.js';

// Set up environment/global variables
let MEMORY_SIZE = 500;
let GRAVITY = 0.005;
let NUM_SIMULATIONS = 3;
let RAND_ACTION_PROB = 0.0;

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
    let model = new Model(10, 2, 2, 100); // Currently set to 10 hidden layer nodes, 2 states (drone y, vy), 2 actions (up, down), 100 batch size

    // Create a drone and render its current position
    let drone = new Drone(canvas, ctx);
    drone.renderDrone(ctx);

    // Allocate droneState variable in memory
    let droneState;
    let action;

    // Run NUM_SIMULATIONS simulations
    for(let sims = 0; sims<NUM_SIMULATIONS; sims++){
        // Run the current simulation until drone crashes
        let crashed = false;
        while(!crashed){
            // Saves browser from crashing
            await sleep(0);

            // Get current drone state
            droneState = drone.getState();

            // Choose and perform action
            action = model.chooseAction(droneState, RAND_ACTION_PROB);
            drone.move(action)

            // Draw on canvas updated parameters
            crashed = draw(canvas, ctx, drone, GRAVITY);
        }

        // Reset drone to initial position
        drone.setToMiddle();
    }
}

/**
 * Execute main program execution on page load
 */
window.onload = function(){
    beginExecution();
}