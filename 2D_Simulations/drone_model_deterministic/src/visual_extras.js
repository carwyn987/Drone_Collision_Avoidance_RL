/**
 * Draws the green transparent range on the canvas to show when the 'drone' is receiving positive reward
 * @param {Canvas} canvas Canvas object
 * @param {Object} ctx Canvas context
 * @param {Number} rtb Reward top boundary - top pixel value to turn green
 * @param {Number} rbb Reward bottom boundary - bottom pixel value to turn green
 */
export default function rewardRange(canvas, ctx, rtb, rbb){
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = '#bbfaaf';
    ctx.fillRect(0, rtb, canvas.width, rbb - rtb);
    ctx.globalAlpha = 1;
}