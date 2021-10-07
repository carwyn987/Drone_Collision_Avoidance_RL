export default class droneConstructor{
    constructor() {
        this.x = 500;
        this.y = 300;
        this.vx = 0;
        this.vy = 0;
        this.angle = 0;
        this.vAngle = 0;
        this.rotateSpeed = 2;
    }

    getDroneStateTensor() {
        return tf.tensor2d([[this.x, this.y, this.vx, this.vy, this.angle, this.vAngle, this.rotateSpeed]]);
    }

    update(gravity) {
        // Update drone state
        this.x += this.vx;
        this.y += this.vy;
        this.vy += gravity;
        this.angle += this.vAngle/2;
        this.vAngle /= 1.3;
    }

    updateMove(action) {
        if(action === null){
            return;
        }
        // nothing
        if(action === 0){
            // do nothing
        }
        // up
        if(action === 1){
            this.vy -= 5*Math.cos(this.angle);
            this.vx += 5*Math.sin(this.angle);
        }
        // down
        if(action === 2){
            this.vy += 5*Math.cos(this.angle);
            this.vx -= 5*Math.sin(this.angle);
        }
        // left
        if(action === 3){
            this.vAngle -= .2;
        }
        // right
        if(action === 4){
            this.vAngle += .2;
        }
    }
}