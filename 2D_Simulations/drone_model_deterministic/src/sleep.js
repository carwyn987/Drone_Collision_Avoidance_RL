/*** This is the sleep function, it describes how long program should timeout for
 * @param {Integer} ms Time in milliseconds program should pause
 * @return {Promise} A promise
 */
export default function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}