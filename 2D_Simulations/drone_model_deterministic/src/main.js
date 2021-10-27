import Drone from './drone.js';
import Ball from './ball.js';
import chooseAction from './model.js';
import draw from './draw.js';
import sleep from './sleep.js';

// Set up environment/global variables
let GRAVITY = 0.01;
let NUM_SIMULATIONS = 99999;
let center = {
    x: 300,
    y: 500
}
let NUM_ACTIONS = 2; // This means choices will be 0,1.

/**
 * Begins execution of main program loop in async function.
 */
async function beginExecution(){
    // Set up canvas (width, height, etc).
    let canvas = document.querySelector("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    let ctx = canvas.getContext('2d');
    
    // Set up variable for number of iterations of training
    let sims = 0;

    // Set up boolean for crash with ball and drone, and counter for red screen
    let droneBallCrash;
    let counter = 0;

    // Create a drone and render its current position
    let drone = new Drone(canvas, ctx);
    drone.renderDrone(ctx);

    // Create a ball and render it
    let ball = new Ball(20,300,2.5,-1.9,30);
    ball.renderBall(ctx);

    // Run NUM_SIMULATIONS simulations
    for( ;sims<NUM_SIMULATIONS; sims++){
        // Run the current simulation until drone crashes
        let crashed = false;
        while(!crashed){
            // Saves browser from crashing
            await sleep(1);

            // Choose and perform action
            let action = chooseAction(drone, ball, center, GRAVITY, NUM_ACTIONS, canvas.height);
            drone.move(action)

            // Draw on canvas updated parameters
            crashed = draw(canvas, ctx, drone, ball, GRAVITY);

            // Check if drone crashed this frame
            droneBallCrash = drone.crashWithBall(ball);

            //If the drone did crash, set background to red and start red counter
            if(droneBallCrash || crashed){
                document.getElementById("canvas").style.backgroundColor = 'red';
                counter = 180;
                if(crashed){
                    await sleep(180);
                }
            }
            // If a recent crash, start counting down
            if(counter!=0){
                counter--;
                if(counter == 1){
                    document.getElementById("canvas").style.backgroundColor = '#e3ffff';
                }
            }

        }

        // Reset counter and background color
        counter = 0;
        document.getElementById("canvas").style.backgroundColor = '#e3ffff';

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