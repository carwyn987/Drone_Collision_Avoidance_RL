/**
 * Calculate the reward the drone at a given position receives
 * @param {Object} drone Drone object with relevant y and vy attributes
 * @param {Number} REWARD_TOP_BOUNDARY The lowest pixel value the drone should receive a + reward
 * @param {Number} REWARD_BOTTOM_BOUNDARY The highest pixel value the drone should receive a + reward
 * @return {Number} Arbitrary reward value (+ if in middle range, - if outside)
 */
export default function calculateReward(drone, REWARD_TOP_BOUNDARY, REWARD_BOTTOM_BOUNDARY){
    if(drone.y >= REWARD_TOP_BOUNDARY && drone.y <= REWARD_BOTTOM_BOUNDARY){
        return 0.5;
    }else{
        return -0.5;
    }
}