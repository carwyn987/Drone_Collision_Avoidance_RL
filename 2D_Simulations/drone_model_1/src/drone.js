export default class droneConstructor{
    constructor() {
        this.x = 500;
        this.y = 300;
        this.vx = 0;
        this.vy = 0;
        this.angle = 0;
        this.vAngle = 0;
        this.rotateSpeed = 2;

        this.scaled = .4;
    }

    getDroneStateTensor() {
        return tf.tensor2d([[this.x, this.y, this.vx, this.vy, this.angle, this.vAngle, this.rotateSpeed]]);
    }

    setRandomPosition() {
        this.x = 500 + Math.round(100*Math.random(),1) - 50;
        this.y = 300;
        this.vx = 0;
        this.vy = 0;
        this.angle = 0;
        this.vAngle = 0;
        this.rotateSpeed = 2;
    }
    

    update(gravity) {
        // Update drone state
        this.x += this.scaled*this.vx;
        this.y += this.scaled*this.vy;
        this.vy += this.scaled*gravity;
        this.angle += this.vAngle/2;
        this.vAngle /= 1.3;
    }

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
        }
        // down
        if(action === 2){
            this.vy += this.scaled*5*Math.cos(this.angle);
            this.vx -= this.scaled*5*Math.sin(this.angle);
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