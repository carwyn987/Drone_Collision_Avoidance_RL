/**
 * Draws the green transparent range on the canvas to show when the 'drone' is receiving positive reward
 * @param {Canvas} canvas Canvas object
 * @param {Object} ctx Canvas context
 * @param {Object} center Canvas center
 */
export default function rewardRange(ctx, center){
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = '#bbfaaf';
    ctx.fillRect(center.x - 10, center.y - 10, 70, 10);
    ctx.globalAlpha = 1;
}