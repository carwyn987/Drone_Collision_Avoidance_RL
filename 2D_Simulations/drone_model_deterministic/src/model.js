import calculateReward from './reward.js';

/**
 * Given a drone state
 * @param {Object} drone Drone object with relevant y and vy attributes
 * @param {Object} ball Ball object with relevant x,y and vx,vy attributes
 * @param {Object} center Ideal point to be at
 * @param {Number} gravity Float representing gravity
 * @param {Number} num_actions Number of actions possible
 * @param {Number} innerHeight height of canvas
 * @return {Number} Value representing best action to take to maximize reward function
 */
export default function chooseAction(drone, ball, center, gravity, num_actions, innerHeight){
    let maxReward = -99999;
    let maxAction = 0;
    for(let i = 0; i<num_actions; i++){
        let curDroneState = drone.getState();
        let curBallState = ball.getState();
        let updateDroneSim = drone.simulateUpdateStateAfterGravity(curDroneState, i, gravity);
        let rewardTemp = calculateReward(updateDroneSim, curBallState, center, innerHeight);
        if(rewardTemp > maxReward){
            maxReward = rewardTemp
            maxAction = i;
        }
    }
    return maxAction;
}