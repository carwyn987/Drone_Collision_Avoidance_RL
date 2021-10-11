import droneConstructor from './drone.js';
import ballConstructor from './ball.js'
import Model from './model.js';
import { Memory } from './memory.js';

export class DroneCanvas{

    constructor() {
        // DEFINE GAME VARIABLES
        this.marginOfIntersection = 10;
        this.gravity = .2;
        this.drone = new droneConstructor(500,300,0,0);
        this.setRandomDronePosition();
        this.ball = new ballConstructor(0,400,10,-10,30);
        this.score = 0;
        this.reward = 0;

        this.addKeyDownHandler(this.drone);
    }

    getDroneBallStateTensor() {
        return tf.concat([this.drone.getDroneStateTensor(), this.ball.getBallStateTensor()], 1);
      }
    
    setRandomDronePosition = function() {
        this.drone.x = 500 + Math.round(100*Math.random(),1) - 50;
        this.drone.y = 300;
        this.drone.vx = 0;
        this.drone.vy = 0;
        this.drone.angle = 0;
        this.drone.vAngle = 0;
        this.drone.rotateSpeed = 2;
    }
    
    droneCrashed = function(droneImg) {
        if(droneImg.height + this.drone.y >= innerHeight){
            return true;
        }
        // Upper boundary of canvas: drone
        if(this.drone.y <= 0){
            return true;
        }
        // Right boundary of canvas: drone
        if(droneImg.width + this.drone.x >= innerWidth){
            return true;
        }
        // Left boundary of canvas: drone
        if(this.drone.x <= 0){
            return true;
        }
    
        // intersection of ball and drone:
        if(this.ball.x + 2 * this.ball.radius - this.marginOfIntersection > this.drone.x && this.ball.x + this.marginOfIntersection < this.drone.x + droneImg.width){
            if(this.ball.y + 2 * this.ball.radius > this.drone.y + this.marginOfIntersection && this.ball.y < this.drone.y + droneImg.height - this.marginOfIntersection){
                return true;
            }
        }
        return false;
    }

    computeReward() {
        // Reward needs to increase over time to reinforce staying alive? maybe?
        // Reward needs to want to stay at a specific point (500,300) or stay away from borders
        // Reward needs to want to stay away from ball
        let distanceFromBall = ((this.drone.y - this.ball.y)**2 + (this.drone.x - this.ball.x)**2)**(1/2);
        let distanceFromPoint = ((this.drone.y - 500)**2 + (this.drone.x - 300)**2)**(1/2);
        
        this.reward = distanceFromBall + (600 - distanceFromPoint);
        if(this.droneCrashed === true){
            this.reward = -200
        }
        this.reward /= 10;
        this.reward = Math.round(this.reward);
        return this.reward;
    }

    addKeyDownHandler(drone){
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
        });
    }
    
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
            this.ball.y = Math.random()*innerHeight;
            this.ball.vx = Math.random()*20 + 10;
            this.ball.vy = Math.random()*10 - 5;
    
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

        // update reward
        this.reward++;
        rewardVal.innerHTML = this.reward;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function run(droneCanvas, l, model) {
    let state = droneCanvas.getDroneBallStateTensor();
    let done = false;
    let eps = 0.1;
    let memoryLength = 500;
    let action = null;
    let reward = null;
    let memory = new Memory(memoryLength);
    let numGames = 300;

    for (let i = 0; i < numGames; ++i) {
        let frameNum = 0;
        while(!done){
            // Render image in browser
            await sleep(20);

            let droneImg = new Image();
            droneImg.src="../images/drone4.png";

            await droneCanvas.draw(l, droneImg);

            done = droneCanvas.droneCrashed(droneImg);

            if(frameNum%6 === 0){
                // Choose action and update move
                action = model.chooseAction(state, eps);
                reward = droneCanvas.computeReward();
                done = droneCanvas.drone.updateMove(action, droneCanvas, droneImg);

                let nextState = droneCanvas.getDroneBallStateTensor();

                memory.addSample([state, action, reward, nextState]);

                state = nextState;
            }

            frameNum++;
        }

        await model.processAndTrain(memory);

        console.log(eps);
        eps -= 0.001
        droneCanvas.drone.setRandomPosition();
        droneCanvas.ball.resetBall(innerHeight);
        done = false;
        // console.log(i);
    }
}

window.onload = function(){
    var canvas = document.querySelector("canvas");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    var l = canvas.getContext('2d');

    let scoreVal = document.getElementById("scoreVal");
    let rewardVal = document.getElementById("rewardVal");

    let droneCanvas = new DroneCanvas();
    let model = new Model(3,12,3,100); //3,12,5,100

    run(droneCanvas, l, model);
}