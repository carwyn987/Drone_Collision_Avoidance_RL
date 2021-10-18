export default class Ball{
    constructor(x,y,vx,vy,radius){
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.radius = radius;
    }

    getBallStateTensor() {
        // return tf.tensor2d([[this.x, this.y, this.vx, this.vy, this.radius]]);
        return tf.tensor2d([[]]);
    }

    update(gravity) {
        // Update ball state - Comment to make ball stop moving
        // this.x += this.vx;
        // this.y += this.vy;
        // this.vy += gravity;
    }

    resetBall(innerHeight) {
        // this.x = 0;
        // this.y = Math.random()*innerHeight;
        // this.vx = Math.random()*20 + 10;
        // this.vy = Math.random()*10 - 5;

        // Uncomment to make ball stop randomly resetting
        this.y = 800
        this.vx = 0
        this.vy = 0
    }
}