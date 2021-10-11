export class Memory {
    constructor(maxMemory) {
        this.maxMemory = maxMemory;
        this.samples = new Array();
    }

    addSample(sample) {
        this.samples.push(sample);
        if (this.samples.length > this.maxMemory){
            this.samples.shift();
        }
    }

    sampleSize(arr, n){
        return arr.slice(0,n);
    }

    sample(nSamples) {
        return this.sampleSize(this.samples, nSamples);
    }
}