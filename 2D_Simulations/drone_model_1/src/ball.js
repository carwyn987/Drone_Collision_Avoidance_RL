export default class Ball{

    /**
     * Ball constructor
     * @param {Integer} x X position of ball
     * @param {Integer} y Y position of ball
     * @param {Float} vx X velocity of ball
     * @param {Float} vy Y velocity of ball
     * @param {Float} radius Radius of ball
     */
    constructor(x,y,vx,vy,radius){
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.radius = radius;
    }

    /**
     * Gets the current attributes of ball and return as a tensor
     * @returns 1x5 tensor of ball attributes
     */
    getBallStateTensor() {
        // Comment to to remove ball from problem
        // return tf.tensor2d([[this.x, this.y, this.vx, this.vy, this.radius]]);

        // Uncomment to to remove ball from problem
        return tf.tensor2d([[]]);
    }

    /**
     * Update ball position and velocity based on gravity value.
     * @param {Float} gravity Value to add to vy per update
     */
    update(gravity) {
        // Update ball state
        // Comment to to remove ball from problem
        // this.x += this.vx;
        // this.y += this.vy;
        // this.vy += gravity;
    }

    /**
     * Reset the ball back to the starting position with random generation of vx, vy, y position.
     * @param {Integer} innerHeight 
     */
    resetBall(innerHeight) {
        // Comment to to remove ball from problem
        this.x = 0;
        // this.y = Math.random()*innerHeight;
        // this.vx = Math.random()*20 + 10;
        // this.vy = Math.random()*10 - 5;

        // Uncomment to to remove ball from problem
        this.y = 800
        this.vx = 0
        this.vy = 0
    }
}