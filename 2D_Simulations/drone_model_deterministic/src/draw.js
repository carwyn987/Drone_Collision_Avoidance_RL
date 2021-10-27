/**
 * Update the canvas based on drone movement, gravity, ball interaction, etc.
 * @param {Object} ctx Context of canvas
 * @param {Object} drone Drone object
 * @param {Object} ball Ball object
 * @return {Boolean} Drone crashed?
 */
export default function draw(canvas, ctx, drone, ball, gravity){
    // Update drone position
    drone.update(gravity);
    ball.update(gravity, canvas);
    
    // Clear canvas
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    // Render drone in new position
    drone.renderDrone(ctx);

    // Update and render ball in canvas
    ball.renderBall(ctx);

    //Check if drone crashed
    let crash = drone.crashed(canvas);

    // If drone crashed, return true
    if(crash){
        return true;
    }

    // Return false : Drone did not crash this iteration
    return false;
}