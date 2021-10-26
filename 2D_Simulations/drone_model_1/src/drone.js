export default class droneConstructor{
    /**
     * Drone object constructor
     */
    constructor() {
        this.x = 500;
        this.y = 300;
        this.vx = 0;
        this.vy = 0;
        this.angle = 0;
        this.vAngle = 0;
        // this.rotateSpeed = 0;
        this.scaled = .4;
        this.maxV = 30;
        this.minV = -30;
    }

    /**
     * Gets the current attributes of drone and return as a tensor
     * @returns 1x7 tensor of drone attributes
     */
    getDroneStateTensor(canvasXMax, canvasYMax) {
        let scaledvx = this.vx/(this.maxV - this.minV) + 0.5;
        let scaledvy = this.vy/(this.maxV - this.minV) + 0.5;
        return tf.tensor2d([[this.x/canvasXMax, this.y/canvasYMax, scaledvx, scaledvy, this.angle/(2*Math.PI), this.vAngle, this.vAngle]]);
    }

    /**
     * Set the drone back to starting position near center of screen. 
     */
    setRandomPosition() {
        this.x = 500 + Math.round(100*Math.random(),1) - 50;
        this.y = 300;
        this.vx = 0;
        this.vy = 0;
        this.angle = 0;
        this.vAngle = 0;
        this.rotateSpeed = 0;
    }
    

    /**
     * Update drone position and velocity based on gravity value.
     * @param {Float} gravity Value to add to vy per update
     */
    update(gravity) {
        // Update drone state
        this.x += this.scaled*this.vx;
        this.y += this.scaled*this.vy;
        this.vy += this.scaled*gravity;
        this.angle += this.vAngle/2;
        this.vAngle /= 1.3;
    }

    /**
     * Move the drone according to input from network choice or randomly chosen action
     * @param {Integer} action Values that associate with an action (nothing, up, down, etc.)
     * @param {Object} droneCanvas Canvas object
     * @param {Image} droneImg Drone image object
     * @return True if drone crashed, false if drone is still in flight
     */
    updateMove(action, droneCanvas, droneImg) {

        // Check if drone crashed
        if(droneCanvas.droneCrashed(droneImg)){
            return true;
        }

        if(action === null){
            return;
        }
        // nothing
        if(action === 0){
            // do nothing
        }
        // up
        if(action === 1){
            this.vy -= this.scaled*5*Math.cos(this.angle);
            this.vx += this.scaled*5*Math.sin(this.angle);
            if(this.vy > this.maxV){
                this.vy = this.maxV;
            }else if(this.vx > this.maxV){
                this.vx = this.maxV;
            }else if(this.vy < this.minV){
                this.vy = this.minV;
            }else if(this.vx < this.minV){
                this.vx = this.minV;
            }
        }
        // down
        if(action === 2){
            this.vy += this.scaled*5*Math.cos(this.angle);
            this.vx -= this.scaled*5*Math.sin(this.angle);
            if(this.vy > this.maxV){
                this.vy = this.maxV;
            }else if(this.vx > this.maxV){
                this.vx = this.maxV;
            }else if(this.vy < this.minV){
                this.vy = this.minV;
            }else if(this.vx < this.minV){
                this.vx = this.minV;
            }
        }
        // left
        if(action === 3){
            this.vAngle -= this.scaled*.2;
        }
        // right
        if(action === 4){
            this.vAngle += this.scaled*.2;
        }
        return false;
    }
}