export default class Memory {
    /**
     * Initialize an array to contain memory samples, as well as the number of elements to store in the array.
     * @param {Integer} maxMemory Max size of memory array
     */ 
     constructor(maxMemory) {
         this.maxMemory = maxMemory;
         this.samples = new Array();
     }
 
     /**
      * Add a sample to the END of memory array
      * If the array gets larger than maxMemory, the oldest sample (at index 0) is discarded
      * 
      * The format of a sample is: [state, action, reward]
      * @param {Object} sample Sample to add
      */
     addSample(sample) {
         this.samples.push(sample);
         if (this.samples.length > this.maxMemory){
             this.samples.shift();
         }
     }
 
     /**
      * Return the most recent frames of the memory
      * These will be the rightmost elements of the array
      * Note: [<-- most dated sample logged --- most recent sample logged -->]
      * @param {Integer} n Number of frames to return
      * @return {Array} An array of n most recent samples logged
      */
     sample(n) {
         return this.samples.slice(-n);
     }
 }