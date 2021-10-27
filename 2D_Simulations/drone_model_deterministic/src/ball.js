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
        return [this.x, this.y, this.vx, this.vy];
    }

    /**
     * Update ball position and velocity based on gravity value.
     * @param {Float} gravity Value to add to vy per update
     */
    update(gravity, canvas) {
        // Update ball state
        this.x += this.vx;
        this.y += this.vy;
        this.vy += gravity;

        // Check if bottom of drone crossed bottom of canvas
        if(this.y >= canvas.height){
            this.resetBall(canvas.height)
        }
        // Check if top of drone crossed top of canvas
        if(this.y <= 0){
            this.resetBall(canvas.height)
        }
    }

    /**
     * Reset the ball back to the starting position with random generation of vx, vy, y position.
     * @param {Integer} innerHeight 
     */
    resetBall(innerHeight) {
        this.x = 0;
        this.y = innerHeight/2 - Math.random()*innerHeight/4;
        this.vx = Math.random()*2 + 3;
        this.vy = Math.random()*2 - 2;
    }

    /**
     * Gets the current attributes of ball and return as an array
     * @returns {tensor2d} 2 attribute tensor of ball y and vy
     */
    getState() {
        return {
            x: this.x,
            y: this.y,
            vx: this.vx,
            vy: this.vy
        }
    }

    /**
     * Renders the ball at the current parameters of referenced ball object
     * @param {Object} ctx Context of the canvas. Used to allow ball class to render ball.
     */
    renderBall(ctx){
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, 2*Math.PI);
        ctx.stroke();
    }
}