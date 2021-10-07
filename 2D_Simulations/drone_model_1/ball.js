export default class Ball{
    constructor(x,y,vx,vy,radius){
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.radius = radius;
    }

    getBallStateTensor() {
        return tf.tensor2d([[this.x, this.y, this.vx, this.vy, this.radius]]);
    }

    update(gravity) {
        // Update ball state
        this.x += this.vx;
        this.y += this.vy;
        this.vy += gravity;
    }
}