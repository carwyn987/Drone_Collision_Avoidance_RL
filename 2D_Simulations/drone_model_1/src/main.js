import droneConstructor from './drone.js';
import ballConstructor from './ball.js'
import Model from './model.js';
import { Memory } from './memory.js';

export class DroneCanvas{

    /**
     * Create a Ball, Drone, and all canvas variables.
     */
    constructor() {
        // DEFINE GAME VARIABLES
        this.marginOfIntersection = 10;
        this.gravity = .2;
        this.drone = new droneConstructor(500,300,0,0);
        this.ball = new ballConstructor(0,400,10,-10,30);
        this.score = 0;
        this.droneImg;
        this.pause = false;

        // Add user input functionality
        this.addKeyDownHandler(this, this.drone);
    }

    /**
     * Get the current state of the environment. This includes the ball and drone position, velocities, and angle/angular velocities.
     * @return A 12x1 array of drone and ball attributes.
     */
    getDroneBallStateTensor(canvasXMax, canvasYMax) {
        return tf.concat([this.drone.getDroneStateTensor(canvasXMax, canvasYMax), this.ball.getBallStateTensor(canvasXMax, canvasYMax)], 1);
      }
    
    /**
     * Sets the drone to a random position near the center of the screen.
     */
    setRandomDronePosition = function() {
        this.drone.x = 500 + Math.round(100*Math.random(),1) - 50;
        this.drone.y = 300;
        this.drone.vx = 0;
        this.drone.vy = 0;
        this.drone.angle = 0;
        this.drone.vAngle = 0;
        this.drone.rotateSpeed = 2;
    }
    
    /**
     * Checks whether or not the drone is currently intersecting the ball or a border of the canvas.
     * @return Boolean
     */
    droneCrashed = function() {
        if(this.droneImg.height + this.drone.y >= innerHeight){
            return true;
        }
        // Upper boundary of canvas: drone
        if(this.drone.y <= 0){
            return true;
        }
        // Right boundary of canvas: drone
        if(this.droneImg.width + this.drone.x >= innerWidth){
            return true;
        }
        // Left boundary of canvas: drone
        if(this.drone.x <= 0){
            return true;
        }
    
        // intersection of ball and drone:
        if(this.ball.x + 2 * this.ball.radius - this.marginOfIntersection > this.drone.x && this.ball.x + this.marginOfIntersection < this.drone.x + this.droneImg.width){
            if(this.ball.y + 2 * this.ball.radius > this.drone.y + this.marginOfIntersection && this.ball.y < this.drone.y + this.droneImg.height - this.marginOfIntersection){
                return true;
            }
        }
        return false;
    }

    /**
     * Compute and return a value representing the goodness of a position
     * @return Integer representing drone location in comparison with where we want it to be.
     */
    computeReward() {
        // Reward needs to increase over time to reinforce staying alive? maybe?
        // Reward needs to want to stay at a specific point (500,300) or stay away from borders
        // Reward needs to want to stay away from ball

        let reward;

        if(this.drone.y <= 250){
            reward = -100;
        }else if(this.drone.y > 250 && this.drone.y <= 350){
            reward = 100;
        }else if(this.drone.y > 350){
            reward = -100;
        }

        if(this.droneCrashed(this.droneImg)){
            reward = -100;
        }


        return reward;
    }

    /**
     * Add key listeners so user can control drone movement and stop the game.
     * 
     * @param {Object} droneCanvas The game object
     * @param {Object} drone The drone object
     */
    addKeyDownHandler(droneCanvas, drone){
        window.addEventListener("keydown", function(event){
                //left
                if(event.keyCode == 37){
                    drone.vAngle -= .2;
                }
                //up
                if(event.keyCode == 38){
                    drone.vy -= 5*Math.cos(drone.angle);
                    drone.vx += 5*Math.sin(drone.angle);
                }
                //right
                if(event.keyCode == 39){
                    drone.vAngle += .2;
                }
                //down
                if(event.keyCode == 40){
                    drone.vy += 5*Math.cos(drone.angle);
                    drone.vx -= 5*Math.sin(drone.angle);
                }
                //space
                if(event.keyCode == 32){
                    droneCanvas.pause = true;
                }
        });
    }
    
    /**
     * Function responsible for visualizing drone and updating canvas based on gravity and input.
     * 
     * @param {Object} l Canvas object
     * @param {Image} droneImg Drone image object
     */
    draw = function(l, droneImg){
        l.clearRect(0, 0, innerWidth, innerHeight);
    
        l.save();
        l.translate( this.drone.x+droneImg.width/2, this.drone.y+droneImg.height/2 );
        l.rotate( this.drone.angle );
        l.translate( -this.drone.x-droneImg.width/2, -this.drone.y-droneImg.height/2 );
        l.drawImage(droneImg, this.drone.x, this.drone.y);
        l.restore();
    
        l.beginPath();
        l.arc(this.ball.x, this.ball.y, this.ball.radius, 0, 2 * Math.PI);
        l.fillStyle = "red";
        l.fill();
        l.stroke();

        this.drone.update(this.gravity);
        this.ball.update(this.gravity);
    
        // Lower boundary of canvas: drone
        if(droneImg.height + this.drone.y >= innerHeight){
            this.drone.vy = -1;
        }
        // Upper boundary of canvas: drone
        if(this.drone.y <= 0){
            this.drone.vy = 1;
        }
        // Right boundary of canvas: drone
        if(droneImg.width + this.drone.x >= innerWidth){
            this.drone.vx = 0;
        }
        // Left boundary of canvas: drone
        if(this.drone.x <= 0){
            this.drone.vx = 0;
        }
        // boundary of canvas: ball
        if(this.ball.y > 200+innerHeight || this.ball.y < -30 || this.ball.x > 100+innerWidth || this.ball.x < -30){
            this.ball.x = 0;
            // Commented out to stop ball from randomizing
            // this.ball.y = Math.random()*innerHeight;
            // this.ball.vx = Math.random()*20 + 10;
            // this.ball.vy = Math.random()*10 - 5;
    
            this.score += 2;
            scoreVal.innerHTML = this.score;
        }
    
        // drone angle calculations (reset angle after flip)
        if(this.drone.angle>2*Math.PI){
            this.drone.angle -= 2*Math.PI;
        }
        if(this.drone.angle<0){
            this.drone.angle += 2*Math.PI;
        }
    
        // intersection of ball and drone:
        if(this.ball.x + 2 * this.ball.radius - this.marginOfIntersection > this.drone.x && this.ball.x + this.marginOfIntersection < this.drone.x + droneImg.width){
            if(this.ball.y + 2 * this.ball.radius > this.drone.y + this.marginOfIntersection && this.ball.y < this.drone.y + droneImg.height - this.marginOfIntersection){
                this.score -= 1;
                scoreVal.innerHTML = this.score;
            }
        }
    }
}

/*** this is the sleep function, it describes sleep
 * @param {Integer} ms time in milliseconds program should pause
 * @return A promise
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Function responsible for running the game in a loop. Main program flow.
 * @param {Object} droneCanvas  The game object
 * @param {Object} l Canvas context
 * @param {Object} model Model object
 */
async function run(droneCanvas, l, model) {
    let state = droneCanvas.getDroneBallStateTensor(innerWidth, innerHeight);
    let done = false;
    let eps = 0.2;
    let memoryLength = 500;
    let action = null;
    let reward = 0;
    let memory = new Memory(memoryLength);
    let numGames = 500;
    let maxFrames = 999;

    let droneImg = new Image();
    droneImg.src="../images/drone4.png";

    droneCanvas.droneImg = droneImg;

    for (let i = 0; i < numGames; ++i) {
        let frameNum = 0;
        while(!done && frameNum < maxFrames){
            // Render image in browser
            await sleep(10);

            if(droneCanvas.pause === true){
                return;
            }

            await droneCanvas.draw(l, droneImg);

            done = droneCanvas.droneCrashed(droneImg);

            // Choose action and update move
            action = model.chooseAction(state, eps);
            reward = droneCanvas.computeReward();
            done = droneCanvas.drone.updateMove(action, droneCanvas, droneImg);

            let nextState = droneCanvas.getDroneBallStateTensor(innerWidth, innerHeight);

            memory.addSample([state, action, reward, nextState]);
            state = nextState;

            frameNum++;
        }

        await model.processAndTrain(memory, frames);

        eps -= 0.005
        droneCanvas.drone.setRandomPosition();
        droneCanvas.ball.resetBall(innerHeight);
        done = false;
    }
}

/**
 * JS Entry point. Begins flow execution, sets up canvas and model. Calls run.
 */
window.onload = function(){
    var canvas = document.querySelector("canvas");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    var l = canvas.getContext('2d');

    let scoreVal = document.getElementById("scoreVal");
    let rewardVal = document.getElementById("rewardVal");

    let droneCanvas = new DroneCanvas();
    let model = new Model(30,7,3,30); //6,12,5,100

    run(droneCanvas, l, model);
}