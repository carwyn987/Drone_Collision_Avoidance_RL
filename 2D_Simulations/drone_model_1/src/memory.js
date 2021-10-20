export class Memory {
   /**
    * Initialize an array to contain memory samples, as well as the number of elements to store in the array.
    * @param {Integer} maxMemory Max size of memory array
    */ 
    constructor(maxMemory) {
        this.maxMemory = maxMemory;
        this.samples = new Array();
    }

    /**
     * Add a sample to the memory array
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
     * @param {Integer} n Number of frames to return
     */
    sample(nSamples) {
        return this.samples.slice(0,nSamples);
    }
}