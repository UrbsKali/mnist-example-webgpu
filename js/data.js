const IMAGE_SIZE = 28 * 28;
const NUM_CLASSES = 10;
const TOTAL_IMAGES = 60000;
const SPRITE_COLS = 600; // Known sprite layout: 600 columns of digits.

const IMAGE_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const LABEL_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export class MNISTData {
  constructor() {
    this.trainImages = null;
    this.trainLabels = null;
    this.testImages = null;
    this.testLabels = null;
    this.trainSize = 0;
    this.testSize = 0;
  }

  async load({ trainSamples, testSamples }, onProgress = () => {}) {
    onProgress(0.02, 'Fetching MNIST labels');
    const labelsResponse = await fetch(LABEL_URL);
    const labelsBuffer = await labelsResponse.arrayBuffer();
    const labelBytes = new Uint8Array(labelsBuffer);

    onProgress(0.15, 'Fetching MNIST sprite image');
    const imageBlob = await fetch(IMAGE_URL).then((res) => res.blob());
    const imageBitmap = await createImageBitmap(imageBlob);

    onProgress(0.25, 'Decoding sprite pixels');
    const canvas = document.createElement('canvas');
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageBitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;

    const allImages = new Float32Array(TOTAL_IMAGES * IMAGE_SIZE);
    const allLabels = new Float32Array(TOTAL_IMAGES * NUM_CLASSES);

    for (let i = 0; i < TOTAL_IMAGES; i += 1) {
      const spriteRow = Math.floor(i / SPRITE_COLS);
      const spriteCol = i % SPRITE_COLS;
      const baseX = spriteCol * 28;
      const baseY = spriteRow * 28;
      const label = labelBytes[i];

      const labelOffset = i * NUM_CLASSES;
      for (let c = 0; c < NUM_CLASSES; c += 1) {
        allLabels[labelOffset + c] = c === label ? 1 : 0;
      }

      const imageOffset = i * IMAGE_SIZE;
      for (let y = 0; y < 28; y += 1) {
        const pixelRow = baseY + y;
        const baseRowIndex = (pixelRow * canvas.width + baseX) * 4;
        for (let x = 0; x < 28; x += 1) {
          const pixelIndex = baseRowIndex + x * 4;
          const pixelValue = pixels[pixelIndex]; // Grayscale sprite (R channel).
          allImages[imageOffset + y * 28 + x] = pixelValue / 255;
        }
      }

      if (i % 1000 === 0) {
        onProgress(0.25 + (i / TOTAL_IMAGES) * 0.6, `Parsing image ${i + 1} / ${TOTAL_IMAGES}`);
      }
    }

    this.trainSize = Math.min(trainSamples, TOTAL_IMAGES - testSamples);
    this.testSize = Math.min(testSamples, TOTAL_IMAGES - this.trainSize);

    this.trainImages = allImages.slice(0, this.trainSize * IMAGE_SIZE);
    this.trainLabels = allLabels.slice(0, this.trainSize * NUM_CLASSES);
    this.testImages = allImages.slice(this.trainSize * IMAGE_SIZE, (this.trainSize + this.testSize) * IMAGE_SIZE);
    this.testLabels = allLabels.slice(this.trainSize * NUM_CLASSES, (this.trainSize + this.testSize) * NUM_CLASSES);

    onProgress(0.95, 'Finalizing dataset');

    // Clean up the temporary canvas to release memory.
    canvas.width = canvas.height = 0;

    onProgress(1, 'Dataset ready');
  }

  getTrainBatch(batchSize, batchIndex, targetImages = new Float32Array(batchSize * IMAGE_SIZE), targetLabels = new Float32Array(batchSize * NUM_CLASSES)) {
    const offset = batchIndex * batchSize;
    const actualBatch = Math.min(batchSize, this.trainSize - offset);
    targetImages.set(this.trainImages.subarray(offset * IMAGE_SIZE, (offset + actualBatch) * IMAGE_SIZE));
    targetLabels.set(this.trainLabels.subarray(offset * NUM_CLASSES, (offset + actualBatch) * NUM_CLASSES));
    return { images: targetImages, labels: targetLabels, size: actualBatch };
  }

  getTestBatch(batchSize, batchIndex, targetImages = new Float32Array(batchSize * IMAGE_SIZE), targetLabels = new Float32Array(batchSize * NUM_CLASSES)) {
    const offset = batchIndex * batchSize;
    const actualBatch = Math.min(batchSize, this.testSize - offset);
    targetImages.set(this.testImages.subarray(offset * IMAGE_SIZE, (offset + actualBatch) * IMAGE_SIZE));
    targetLabels.set(this.testLabels.subarray(offset * NUM_CLASSES, (offset + actualBatch) * NUM_CLASSES));
    return { images: targetImages, labels: targetLabels, size: actualBatch };
  }
}

export const DataConstants = {
  imageSize: IMAGE_SIZE,
  numClasses: NUM_CLASSES,
  height: 28,
  width: 28,
};
