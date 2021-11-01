/**
 * Update the canvas based on drone movement, gravity, ball interaction, etc.
 * @param {Object} ctx Context of canvas
 * @param {Object} drone Drone object
 * @return {Boolean} Drone crashed?
 */
export default function draw(canvas, ctx, drone, gravity){
    // Update drone position
    drone.update(gravity);
    
    // Clear canvas
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    // Render drone in new position
    drone.renderDrone(ctx);

    //Check if drone crashed
    let crash = drone.crashed(canvas);

    // If drone crashed, return true
    if(crash){
        return true;
    }

    // Return false : Drone did not crash this iteration
    return false;
}