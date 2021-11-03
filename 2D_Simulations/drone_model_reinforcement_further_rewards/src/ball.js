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
     * @param {Canvas} canvas Canvas object to get height from
     * @returns {tensor2d} 2 attribute tensor of ball y and vy
     */
    getState(canvas) {
        // First, scale the y value
        let yScaled = this.y/(canvas.height)-0.5;
        // Now the x value
        let xScaled = this.x/(canvas.width)-0.5;

        // Now scale vy value
        // Let the max range be [-10, 10]. Therefore to transform, divide by 20 to reduce to [-0.5, 0.5]
        let vyScaled = (this.vy/20);
        // Now scale vx value
        // Let the max range be [-5, 5]. Therefore to transform, divide by 10 to reduce to [-0.5, 0.5]
        let vxScaled = (this.vx/20);
        
        return tf.tensor2d([[xScaled, yScaled, vxScaled, vyScaled]]);
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

        // Check if ball passed drone
        if(this.x > canvas.width/2 + 160){
            this.resetBall(canvas.height)
        }
    }

    /**
     * Reset the ball back to the starting position with random generation of vx, vy, y position.
     * @param {Integer} innerHeight 
     */
    resetBall(innerHeight) {
        this.x = 0;
        // this.y = innerHeight/2 - Math.random()*innerHeight/4;
        // this.vx = Math.random()*2 + 4;
        // this.vy = Math.random()*2 - 2;
        this.y = innerHeight/2;
        this.vx = 5;
        this.vy = -1.8;
    }

    /**
     * Renders the ball at the current parameters of referenced ball object
     * @param {Object} ctx Context of the canvas. Used to allow ball class to render ball.
     */
    renderBall(ctx){
        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, 2*Math.PI);
        ctx.stroke();
        ctx.fill();
    }
}