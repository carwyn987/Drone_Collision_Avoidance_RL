/**
 * Calculate the reward the drone at a given position receives
 * @param {Object} droneObj Drone object with relevant y and vy attributes
 * @param {Object} ballObj Ball object with relevant x,y and vx,vy attributes
 * @param {Array} center, contains x and y attributes for where the ideal position is
 * @param {Number} innerHeight height of canvas
 * @return {Number} reward value
 */
export default function calculateReward(droneObj, ballObj, center, innerHeight){
    let distanceFromCenter = ((droneObj.x - center.x)**2 + (droneObj.y - center.y)**2)**(1/2);
    let distanceFromBall = ((ballObj.x - droneObj.x)**2 + (ballObj.y - droneObj.y)**2)**(1/2);
    let distanceFromEdge = 200/(droneObj.y - 0 < innerHeight -droneObj.y ? droneObj.y - 0 : innerHeight -droneObj.y);
    return (1000 - distanceFromCenter) - 30*distanceFromEdge - 20000/distanceFromBall - 0.1*Math.abs(droneObj.vy);
    
    // Comment out above return statement and uncomment this line to visualize crashes
    // return 1000 - distanceFromCenter;
}