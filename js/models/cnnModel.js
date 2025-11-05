import { loadShaderModule, createBuffer, createEmptyBuffer, writeBuffer, readBufferToArray, ceilDiv } from '../wgpu.js';
import { DataConstants } from '../data.js';

const WORKGROUP_128 = 128;
const WORKGROUP_64 = 64;
const WORKGROUP_32 = 32;

const INPUT_WIDTH = 28;
const INPUT_HEIGHT = 28;
const INPUT_CHANNELS = 1;
const CONV_OUT_CHANNELS = 8;
const CONV_KERNEL = 5;
const POOL_SIZE = 2;
const POOL_STRIDE = 2;
const POOL_WIDTH = INPUT_WIDTH / POOL_STRIDE;
const POOL_HEIGHT = INPUT_HEIGHT / POOL_STRIDE;

export class CnnModel {
  constructor(device) {
    this.device = device;
    this.queue = device.queue;

    this.compiled = false;
    this.initialized = false;

    this.optimizer = 'adam';
    this.learningRate = 0.001;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.beta1Power = this.beta1;
    this.beta2Power = this.beta2;
    this.step = 0;

    this.classes = DataConstants.numClasses;
    this.imageSize = DataConstants.imageSize;
    this.convWeightCount = CONV_KERNEL * CONV_KERNEL * INPUT_CHANNELS * CONV_OUT_CHANNELS;
    this.convOutputElementsPerSample = INPUT_WIDTH * INPUT_HEIGHT * CONV_OUT_CHANNELS;
    this.poolOutputElementsPerSample = POOL_WIDTH * POOL_HEIGHT * CONV_OUT_CHANNELS;
    this.denseFeatureCount = this.poolOutputElementsPerSample;
    this.denseWeightCount = this.denseFeatureCount * this.classes;

    this.uniformBuffers = {};
    this.bindGroups = {};
    this.zeroBindGroups = new Map();
    this.pipelines = {};
  }

  async compile() {
    if (this.compiled) {
      return;
    }

    const [
      convModule,
      reluFModule,
      reluBModule,
      poolFModule,
      poolBModule,
      flattenModule,
      unflattenModule,
      denseModule,
      softmaxModule,
      gradDenseWModule,
      gradDenseInputModule,
      reduceModule,
      scaleModule,
      sgdModule,
      adamModule,
      zeroModule,
      accuracyModule,
      convFilterModule,
    ] = await Promise.all([
      loadShaderModule(this.device, 'shaders/conv2d_forward.wgsl'),
      loadShaderModule(this.device, 'shaders/relu_forward.wgsl'),
      loadShaderModule(this.device, 'shaders/relu_backward.wgsl'),
      loadShaderModule(this.device, 'shaders/maxpool_forward.wgsl'),
      loadShaderModule(this.device, 'shaders/maxpool_backward.wgsl'),
      loadShaderModule(this.device, 'shaders/flatten.wgsl'),
      loadShaderModule(this.device, 'shaders/unflatten.wgsl'),
      loadShaderModule(this.device, 'shaders/matmul_bias.wgsl'),
      loadShaderModule(this.device, 'shaders/softmax_cross_entropy_grad.wgsl'),
      loadShaderModule(this.device, 'shaders/matmul_at_b.wgsl'),
      loadShaderModule(this.device, 'shaders/matmul_abt.wgsl'),
      loadShaderModule(this.device, 'shaders/reduce_sum_axis0.wgsl'),
      loadShaderModule(this.device, 'shaders/scale_buffer.wgsl'),
      loadShaderModule(this.device, 'shaders/sgd_update.wgsl'),
      loadShaderModule(this.device, 'shaders/adam_update.wgsl'),
      loadShaderModule(this.device, 'shaders/zero_buffer.wgsl'),
      loadShaderModule(this.device, 'shaders/argmax_accuracy.wgsl'),
      loadShaderModule(this.device, 'shaders/conv2d_backprop_filter.wgsl'),
    ]);

    this.pipelines.convForward = this.device.createComputePipeline({
      label: 'cnn-conv-forward',
      layout: 'auto',
      compute: { module: convModule, entryPoint: 'main' },
    });
    this.pipelines.reluForward = this.device.createComputePipeline({
      label: 'cnn-relu-forward',
      layout: 'auto',
      compute: { module: reluFModule, entryPoint: 'main' },
    });
    this.pipelines.reluBackward = this.device.createComputePipeline({
      label: 'cnn-relu-backward',
      layout: 'auto',
      compute: { module: reluBModule, entryPoint: 'main' },
    });
    this.pipelines.poolForward = this.device.createComputePipeline({
      label: 'cnn-pool-forward',
      layout: 'auto',
      compute: { module: poolFModule, entryPoint: 'main' },
    });
    this.pipelines.poolBackward = this.device.createComputePipeline({
      label: 'cnn-pool-backward',
      layout: 'auto',
      compute: { module: poolBModule, entryPoint: 'main' },
    });
    this.pipelines.flatten = this.device.createComputePipeline({
      label: 'cnn-flatten',
      layout: 'auto',
      compute: { module: flattenModule, entryPoint: 'main' },
    });
    this.pipelines.unflatten = this.device.createComputePipeline({
      label: 'cnn-unflatten',
      layout: 'auto',
      compute: { module: unflattenModule, entryPoint: 'main' },
    });
    this.pipelines.denseForward = this.device.createComputePipeline({
      label: 'cnn-dense-forward',
      layout: 'auto',
      compute: { module: denseModule, entryPoint: 'main' },
    });
    this.pipelines.softmax = this.device.createComputePipeline({
      label: 'cnn-softmax',
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'main' },
    });
    this.pipelines.gradDenseWeights = this.device.createComputePipeline({
      label: 'cnn-grad-denseW',
      layout: 'auto',
      compute: { module: gradDenseWModule, entryPoint: 'main' },
    });
    this.pipelines.gradDenseInput = this.device.createComputePipeline({
      label: 'cnn-grad-denseInput',
      layout: 'auto',
      compute: { module: gradDenseInputModule, entryPoint: 'main' },
    });
    this.pipelines.reduce = this.device.createComputePipeline({
      label: 'cnn-reduce',
      layout: 'auto',
      compute: { module: reduceModule, entryPoint: 'main' },
    });
    this.pipelines.scale = this.device.createComputePipeline({
      label: 'cnn-scale',
      layout: 'auto',
      compute: { module: scaleModule, entryPoint: 'main' },
    });
    this.pipelines.sgd = this.device.createComputePipeline({
      label: 'cnn-sgd',
      layout: 'auto',
      compute: { module: sgdModule, entryPoint: 'main' },
    });
    this.pipelines.adam = this.device.createComputePipeline({
      label: 'cnn-adam',
      layout: 'auto',
      compute: { module: adamModule, entryPoint: 'main' },
    });
    this.pipelines.zero = this.device.createComputePipeline({
      label: 'cnn-zero',
      layout: 'auto',
      compute: { module: zeroModule, entryPoint: 'main' },
    });
    this.pipelines.accuracy = this.device.createComputePipeline({
      label: 'cnn-accuracy',
      layout: 'auto',
      compute: { module: accuracyModule, entryPoint: 'main' },
    });
    this.pipelines.convFilterGrad = this.device.createComputePipeline({
      label: 'cnn-conv-filter-grad',
      layout: 'auto',
      compute: { module: convFilterModule, entryPoint: 'main' },
    });

    this.compiled = true;
  }

  createUniformBuffer(byteLength, label) {
    const aligned = Math.ceil(byteLength / 16) * 16;
    return this.device.createBuffer({
      label,
      size: aligned,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  async initialize({ batchSize, learningRate, optimizer, seed }) {
    await this.compile();

    this.batchSize = batchSize;
    this.learningRate = learningRate;
    this.optimizer = optimizer;

    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.epsilon = 1e-8;
    this.beta1Power = this.beta1;
    this.beta2Power = this.beta2;
    this.step = 0;

    const rng = mulberry32(seed ?? 42);

    const convWeights = new Float32Array(this.convWeightCount);
    const convFanIn = CONV_KERNEL * CONV_KERNEL * INPUT_CHANNELS;
    const convFanOut = CONV_KERNEL * CONV_KERNEL * CONV_OUT_CHANNELS;
    const convStd = Math.sqrt(2 / (convFanIn + convFanOut));
    for (let i = 0; i < convWeights.length; i += 1) {
      convWeights[i] = (rng() * 2 - 1) * convStd;
    }
    const convBias = new Float32Array(CONV_OUT_CHANNELS);

    const denseWeights = new Float32Array(this.denseWeightCount);
    const denseStd = Math.sqrt(2 / (this.denseFeatureCount + this.classes));
    for (let i = 0; i < denseWeights.length; i += 1) {
      denseWeights[i] = (rng() * 2 - 1) * denseStd;
    }
    const denseBias = new Float32Array(this.classes);

    this.convWeightBuffer = createBuffer(this.device, convWeights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'cnn-convW');
    this.convBiasBuffer = createBuffer(this.device, convBias, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'cnn-convB');
    this.denseWeightBuffer = createBuffer(this.device, denseWeights, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'cnn-denseW');
    this.denseBiasBuffer = createBuffer(this.device, denseBias, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'cnn-denseB');

    this.gradConvWeightBuffer = createEmptyBuffer(this.device, convWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-grad-convW');
    this.gradConvBiasBuffer = createEmptyBuffer(this.device, convBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-grad-convB');
    this.gradDenseWeightBuffer = createEmptyBuffer(this.device, denseWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-grad-denseW');
    this.gradDenseBiasBuffer = createEmptyBuffer(this.device, denseBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-grad-denseB');

    this.mConvWeightBuffer = createEmptyBuffer(this.device, convWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-m-convW');
    this.vConvWeightBuffer = createEmptyBuffer(this.device, convWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-v-convW');
    this.mConvBiasBuffer = createEmptyBuffer(this.device, convBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-m-convB');
    this.vConvBiasBuffer = createEmptyBuffer(this.device, convBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-v-convB');
    this.mDenseWeightBuffer = createEmptyBuffer(this.device, denseWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-m-denseW');
    this.vDenseWeightBuffer = createEmptyBuffer(this.device, denseWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-v-denseW');
    this.mDenseBiasBuffer = createEmptyBuffer(this.device, denseBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-m-denseB');
    this.vDenseBiasBuffer = createEmptyBuffer(this.device, denseBias.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-v-denseB');

    const inputByteSize = this.batchSize * this.imageSize * 4;
    const convOutputByteSize = this.batchSize * this.convOutputElementsPerSample * 4;
    const poolOutputByteSize = this.batchSize * this.poolOutputElementsPerSample * 4;
    const poolMaskByteSize = this.batchSize * this.poolOutputElementsPerSample * 4;
    const denseInputByteSize = this.batchSize * this.denseFeatureCount * 4;
    const logitsByteSize = this.batchSize * this.classes * 4;

    this.inputBuffer = createEmptyBuffer(this.device, inputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-input');
    this.labelBuffer = createEmptyBuffer(this.device, this.batchSize * this.classes * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'cnn-labels');
    this.convOutputBuffer = createEmptyBuffer(this.device, convOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-convOut');
    this.reluOutputBuffer = createEmptyBuffer(this.device, convOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-reluOut');
    this.poolOutputBuffer = createEmptyBuffer(this.device, poolOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-poolOut');
    this.poolMaskBuffer = createEmptyBuffer(this.device, poolMaskByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-poolMask');
    this.flattenBuffer = createEmptyBuffer(this.device, denseInputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-flatten');
    this.logitsBuffer = createEmptyBuffer(this.device, logitsByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-logits');
    this.probBuffer = createEmptyBuffer(this.device, logitsByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-probs');
    this.gradLogitsBuffer = createEmptyBuffer(this.device, logitsByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-gradLogits');
    this.lossBuffer = createEmptyBuffer(this.device, this.batchSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-loss');
    this.accuracyMaskBuffer = createEmptyBuffer(this.device, this.batchSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-acc-mask');

    this.gradDenseInputBuffer = createEmptyBuffer(this.device, denseInputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-gradDenseInput');
    this.gradPoolOutputBuffer = createEmptyBuffer(this.device, poolOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-gradPoolOut');
    this.gradReluOutputBuffer = createEmptyBuffer(this.device, convOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-gradReluOut');
    this.gradConvOutputBuffer = createEmptyBuffer(this.device, convOutputByteSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'cnn-gradConvOut');

    this.uniformBuffers.conv = this.createUniformBuffer(32, 'cnn-u-conv');
    this.uniformBuffers.poolForward = this.createUniformBuffer(32, 'cnn-u-poolF');
    this.uniformBuffers.poolBackward = this.createUniformBuffer(16, 'cnn-u-poolB');
    this.uniformBuffers.flatten = this.createUniformBuffer(32, 'cnn-u-flatten');
    this.uniformBuffers.unflatten = this.createUniformBuffer(32, 'cnn-u-unflatten');
    this.uniformBuffers.denseForward = this.createUniformBuffer(16, 'cnn-u-dense');
    this.uniformBuffers.softmax = this.createUniformBuffer(16, 'cnn-u-softmax');
    this.uniformBuffers.gradDenseWeights = this.createUniformBuffer(16, 'cnn-u-gradDenseW');
    this.uniformBuffers.backDense = this.createUniformBuffer(16, 'cnn-u-backDense');
    this.uniformBuffers.reduceDenseBias = this.createUniformBuffer(16, 'cnn-u-redDenseB');
    this.uniformBuffers.reduceConvBias = this.createUniformBuffer(16, 'cnn-u-redConvB');
    this.uniformBuffers.scaleDenseWeights = this.createUniformBuffer(16, 'cnn-u-scaleDenseW');
    this.uniformBuffers.scaleDenseBias = this.createUniformBuffer(16, 'cnn-u-scaleDenseB');
    this.uniformBuffers.scaleConvWeights = this.createUniformBuffer(16, 'cnn-u-scaleConvW');
    this.uniformBuffers.scaleConvBias = this.createUniformBuffer(16, 'cnn-u-scaleConvB');
    this.uniformBuffers.sgdDenseWeights = this.createUniformBuffer(16, 'cnn-u-sgdDenseW');
    this.uniformBuffers.sgdDenseBias = this.createUniformBuffer(16, 'cnn-u-sgdDenseB');
    this.uniformBuffers.sgdConvWeights = this.createUniformBuffer(16, 'cnn-u-sgdConvW');
    this.uniformBuffers.sgdConvBias = this.createUniformBuffer(16, 'cnn-u-sgdConvB');
    this.uniformBuffers.adamDenseWeights = this.createUniformBuffer(48, 'cnn-u-adamDenseW');
    this.uniformBuffers.adamDenseBias = this.createUniformBuffer(48, 'cnn-u-adamDenseB');
    this.uniformBuffers.adamConvWeights = this.createUniformBuffer(48, 'cnn-u-adamConvW');
    this.uniformBuffers.adamConvBias = this.createUniformBuffer(48, 'cnn-u-adamConvB');
    this.uniformBuffers.zero = this.createUniformBuffer(16, 'cnn-u-zero');
    this.uniformBuffers.accuracy = this.createUniformBuffer(16, 'cnn-u-accuracy');

    this.createBindGroups();

    this.initialized = true;
  }

  updateConvUniform(batch) {
    const stride = 1;
    const padding = Math.floor(CONV_KERNEL / 2);
    const info = new Uint32Array([
      INPUT_WIDTH,
      INPUT_HEIGHT,
      INPUT_CHANNELS,
      CONV_OUT_CHANNELS,
      CONV_KERNEL,
      stride,
      padding,
      batch,
    ]);
    this.queue.writeBuffer(this.uniformBuffers.conv, 0, info);
  }

  updatePoolUniform(batch) {
    const info = new Uint32Array([
      INPUT_WIDTH,
      INPUT_HEIGHT,
      CONV_OUT_CHANNELS,
      POOL_SIZE,
      POOL_STRIDE,
      batch,
      POOL_WIDTH,
      POOL_HEIGHT,
    ]);
    this.queue.writeBuffer(this.uniformBuffers.poolForward, 0, info);
  }

  updatePoolBackwardUniform(batch) {
    const total = batch * this.poolOutputElementsPerSample;
    const info = new Uint32Array([total]);
    this.queue.writeBuffer(this.uniformBuffers.poolBackward, 0, info);
  }

  updateFlattenUniform(batch) {
    const info = new Uint32Array([
      POOL_WIDTH,
      POOL_HEIGHT,
      CONV_OUT_CHANNELS,
      this.denseFeatureCount,
      batch,
    ]);
    this.queue.writeBuffer(this.uniformBuffers.flatten, 0, info);
    this.queue.writeBuffer(this.uniformBuffers.unflatten, 0, info);
  }

  updateDenseForwardUniform(batch) {
    const info = new Uint32Array([batch, this.classes, this.denseFeatureCount, 0]);
    this.queue.writeBuffer(this.uniformBuffers.denseForward, 0, info);
  }

  updateSoftmaxUniform(batch) {
    const info = new Float32Array([batch, this.classes, 1e-7, 0]);
    this.queue.writeBuffer(this.uniformBuffers.softmax, 0, info);
  }

  updateGradDenseWeightsUniform(batch) {
    const info = new Uint32Array([batch, this.denseFeatureCount, this.classes, 0]);
    this.queue.writeBuffer(this.uniformBuffers.gradDenseWeights, 0, info);
  }

  updateBackDenseUniform(batch) {
    const info = new Uint32Array([batch, this.denseFeatureCount, this.classes]);
    this.queue.writeBuffer(this.uniformBuffers.backDense, 0, info);
  }

  updateReduceDenseBiasUniform(batch) {
    const info = new Uint32Array([batch, this.classes, 0, 0]);
    this.queue.writeBuffer(this.uniformBuffers.reduceDenseBias, 0, info);
  }

  updateReduceConvBiasUniform(batch) {
    const samples = batch * INPUT_WIDTH * INPUT_HEIGHT;
    const info = new Uint32Array([samples, CONV_OUT_CHANNELS, 0, 0]);
    this.queue.writeBuffer(this.uniformBuffers.reduceConvBias, 0, info);
  }

  updateScaleUniform(buffer, size, factor) {
    const info = new Float32Array([factor, size, 0, 0]);
    this.queue.writeBuffer(buffer, 0, info);
  }

  updateSgdUniform(buffer, size) {
    const info = new Float32Array([this.learningRate, size, 0, 0]);
    this.queue.writeBuffer(buffer, 0, info);
  }

  updateAdamUniform(buffer, size) {
    const info = new Float32Array([
      this.learningRate,
      this.beta1,
      this.beta2,
      this.epsilon,
      1 - this.beta1,
      1 - this.beta2,
      this.beta1Power,
      this.beta2Power,
      size,
    ]);
    this.queue.writeBuffer(buffer, 0, info);
  }

  updateZeroUniform(size) {
    const info = new Uint32Array([size]);
    this.queue.writeBuffer(this.uniformBuffers.zero, 0, info);
  }

  updateAccuracyUniform(batch) {
    const info = new Uint32Array([batch, this.classes, 0, 0]);
    this.queue.writeBuffer(this.uniformBuffers.accuracy, 0, info);
  }

  createBindGroups() {
    this.bindGroups.convForward = this.device.createBindGroup({
      layout: this.pipelines.convForward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.convWeightBuffer } },
        { binding: 2, resource: { buffer: this.convBiasBuffer } },
        { binding: 3, resource: { buffer: this.convOutputBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.conv } },
      ],
    });

    this.bindGroups.reluForward = this.device.createBindGroup({
      layout: this.pipelines.reluForward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convOutputBuffer } },
        { binding: 1, resource: { buffer: this.reluOutputBuffer } },
      ],
    });

    this.bindGroups.poolForward = this.device.createBindGroup({
      layout: this.pipelines.poolForward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.reluOutputBuffer } },
        { binding: 1, resource: { buffer: this.poolOutputBuffer } },
        { binding: 2, resource: { buffer: this.poolMaskBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.poolForward } },
      ],
    });

    this.bindGroups.flatten = this.device.createBindGroup({
      layout: this.pipelines.flatten.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.poolOutputBuffer } },
        { binding: 1, resource: { buffer: this.flattenBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.flatten } },
      ],
    });

    this.bindGroups.denseForward = this.device.createBindGroup({
      layout: this.pipelines.denseForward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.flattenBuffer } },
        { binding: 1, resource: { buffer: this.denseWeightBuffer } },
        { binding: 2, resource: { buffer: this.denseBiasBuffer } },
        { binding: 3, resource: { buffer: this.logitsBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.denseForward } },
      ],
    });

    this.bindGroups.softmax = this.device.createBindGroup({
      layout: this.pipelines.softmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.logitsBuffer } },
        { binding: 1, resource: { buffer: this.labelBuffer } },
        { binding: 2, resource: { buffer: this.probBuffer } },
        { binding: 3, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 4, resource: { buffer: this.lossBuffer } },
        { binding: 5, resource: { buffer: this.uniformBuffers.softmax } },
      ],
    });

    this.bindGroups.gradDenseWeights = this.device.createBindGroup({
      layout: this.pipelines.gradDenseWeights.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.flattenBuffer } },
        { binding: 1, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 2, resource: { buffer: this.gradDenseWeightBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.gradDenseWeights } },
      ],
    });

    this.bindGroups.reduceDenseBias = this.device.createBindGroup({
      layout: this.pipelines.reduce.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 1, resource: { buffer: this.gradDenseBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.reduceDenseBias } },
      ],
    });

    this.bindGroups.scaleGradDenseWeights = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradDenseWeightBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleDenseWeights } },
      ],
    });

    this.bindGroups.scaleGradDenseBias = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradDenseBiasBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleDenseBias } },
      ],
    });

    this.bindGroups.sgdDenseWeights = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.denseWeightBuffer } },
        { binding: 1, resource: { buffer: this.gradDenseWeightBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdDenseWeights } },
      ],
    });

    this.bindGroups.sgdDenseBias = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.denseBiasBuffer } },
        { binding: 1, resource: { buffer: this.gradDenseBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdDenseBias } },
      ],
    });

    this.bindGroups.adamDenseWeights = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.denseWeightBuffer } },
        { binding: 1, resource: { buffer: this.gradDenseWeightBuffer } },
        { binding: 2, resource: { buffer: this.mDenseWeightBuffer } },
        { binding: 3, resource: { buffer: this.vDenseWeightBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamDenseWeights } },
      ],
    });

    this.bindGroups.adamDenseBias = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.denseBiasBuffer } },
        { binding: 1, resource: { buffer: this.gradDenseBiasBuffer } },
        { binding: 2, resource: { buffer: this.mDenseBiasBuffer } },
        { binding: 3, resource: { buffer: this.vDenseBiasBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamDenseBias } },
      ],
    });

    this.bindGroups.gradDenseInput = this.device.createBindGroup({
      layout: this.pipelines.gradDenseInput.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradLogitsBuffer } },
        { binding: 1, resource: { buffer: this.denseWeightBuffer } },
        { binding: 2, resource: { buffer: this.gradDenseInputBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.backDense } },
      ],
    });

    this.bindGroups.unflatten = this.device.createBindGroup({
      layout: this.pipelines.unflatten.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradDenseInputBuffer } },
        { binding: 1, resource: { buffer: this.gradPoolOutputBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.unflatten } },
      ],
    });

    this.bindGroups.poolBackward = this.device.createBindGroup({
      layout: this.pipelines.poolBackward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradPoolOutputBuffer } },
        { binding: 1, resource: { buffer: this.poolMaskBuffer } },
        { binding: 2, resource: { buffer: this.gradReluOutputBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.poolBackward } },
      ],
    });

    this.bindGroups.reluBackward = this.device.createBindGroup({
      layout: this.pipelines.reluBackward.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradReluOutputBuffer } },
        { binding: 1, resource: { buffer: this.reluOutputBuffer } },
        { binding: 2, resource: { buffer: this.gradConvOutputBuffer } },
      ],
    });

    this.bindGroups.convFilterGrad = this.device.createBindGroup({
      layout: this.pipelines.convFilterGrad.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.inputBuffer } },
        { binding: 1, resource: { buffer: this.gradConvOutputBuffer } },
        { binding: 2, resource: { buffer: this.gradConvWeightBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.conv } },
      ],
    });

    this.bindGroups.reduceConvBias = this.device.createBindGroup({
      layout: this.pipelines.reduce.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradConvOutputBuffer } },
        { binding: 1, resource: { buffer: this.gradConvBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.reduceConvBias } },
      ],
    });

    this.bindGroups.scaleGradConvWeights = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradConvWeightBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleConvWeights } },
      ],
    });

    this.bindGroups.scaleGradConvBias = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gradConvBiasBuffer } },
        { binding: 1, resource: { buffer: this.uniformBuffers.scaleConvBias } },
      ],
    });

    this.bindGroups.sgdConvWeights = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convWeightBuffer } },
        { binding: 1, resource: { buffer: this.gradConvWeightBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdConvWeights } },
      ],
    });

    this.bindGroups.sgdConvBias = this.device.createBindGroup({
      layout: this.pipelines.sgd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convBiasBuffer } },
        { binding: 1, resource: { buffer: this.gradConvBiasBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffers.sgdConvBias } },
      ],
    });

    this.bindGroups.adamConvWeights = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convWeightBuffer } },
        { binding: 1, resource: { buffer: this.gradConvWeightBuffer } },
        { binding: 2, resource: { buffer: this.mConvWeightBuffer } },
        { binding: 3, resource: { buffer: this.vConvWeightBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamConvWeights } },
      ],
    });

    this.bindGroups.adamConvBias = this.device.createBindGroup({
      layout: this.pipelines.adam.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.convBiasBuffer } },
        { binding: 1, resource: { buffer: this.gradConvBiasBuffer } },
        { binding: 2, resource: { buffer: this.mConvBiasBuffer } },
        { binding: 3, resource: { buffer: this.vConvBiasBuffer } },
        { binding: 4, resource: { buffer: this.uniformBuffers.adamConvBias } },
      ],
    });

    this.bindGroups.accuracy = this.device.createBindGroup({
      layout: this.pipelines.accuracy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.probBuffer } },
        { binding: 1, resource: { buffer: this.labelBuffer } },
        { binding: 2, resource: { buffer: this.accuracyMaskBuffer } },
        { binding: 3, resource: { buffer: this.uniformBuffers.accuracy } },
      ],
    });
  }

  getZeroBindGroup(buffer) {
    if (!this.zeroBindGroups.has(buffer)) {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.zero.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: this.uniformBuffers.zero } },
        ],
      });
      this.zeroBindGroups.set(buffer, bindGroup);
    }
    return this.zeroBindGroups.get(buffer);
  }

  encodeForwardPass(encoder, batchSize) {
    this.updateConvUniform(batchSize);
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.convForward);
    pass.setBindGroup(0, this.bindGroups.convForward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.convOutputElementsPerSample, WORKGROUP_64));
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.reluForward);
    pass.setBindGroup(0, this.bindGroups.reluForward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.convOutputElementsPerSample, WORKGROUP_128));
    pass.end();

    this.updatePoolUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.poolForward);
    pass.setBindGroup(0, this.bindGroups.poolForward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.poolOutputElementsPerSample, WORKGROUP_64));
    pass.end();

    this.updateFlattenUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.flatten);
    pass.setBindGroup(0, this.bindGroups.flatten);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.denseFeatureCount, WORKGROUP_128));
    pass.end();

    this.updateDenseForwardUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.denseForward);
    pass.setBindGroup(0, this.bindGroups.denseForward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.classes, WORKGROUP_128));
    pass.end();

    this.updateSoftmaxUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.softmax);
    pass.setBindGroup(0, this.bindGroups.softmax);
    pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
    pass.end();
  }

  encodeBackwardPass(encoder, batchSize) {
    this.updateGradDenseWeightsUniform(batchSize);
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.gradDenseWeights);
    pass.setBindGroup(0, this.bindGroups.gradDenseWeights);
    pass.dispatchWorkgroups(ceilDiv(this.denseWeightCount, WORKGROUP_128));
    pass.end();

    this.updateReduceDenseBiasUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.reduce);
    pass.setBindGroup(0, this.bindGroups.reduceDenseBias);
    pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_64));
    pass.end();

    const scaleFactor = 1 / batchSize;
    this.updateScaleUniform(this.uniformBuffers.scaleDenseWeights, this.denseWeightCount, scaleFactor);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.scale);
    pass.setBindGroup(0, this.bindGroups.scaleGradDenseWeights);
    pass.dispatchWorkgroups(ceilDiv(this.denseWeightCount, WORKGROUP_128));
    pass.end();

    this.updateScaleUniform(this.uniformBuffers.scaleDenseBias, this.classes, scaleFactor);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.scale);
    pass.setBindGroup(0, this.bindGroups.scaleGradDenseBias);
    pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
    pass.end();

    this.updateBackDenseUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.gradDenseInput);
    pass.setBindGroup(0, this.bindGroups.gradDenseInput);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.denseFeatureCount, WORKGROUP_128));
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.unflatten);
    pass.setBindGroup(0, this.bindGroups.unflatten);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.poolOutputElementsPerSample, WORKGROUP_128));
    pass.end();

    this.updateZeroUniform(batchSize * this.convOutputElementsPerSample);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.zero);
    pass.setBindGroup(0, this.getZeroBindGroup(this.gradReluOutputBuffer));
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.convOutputElementsPerSample, WORKGROUP_128));
    pass.end();

    this.updatePoolBackwardUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.poolBackward);
    pass.setBindGroup(0, this.bindGroups.poolBackward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.poolOutputElementsPerSample, WORKGROUP_64));
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.reluBackward);
    pass.setBindGroup(0, this.bindGroups.reluBackward);
    pass.dispatchWorkgroups(ceilDiv(batchSize * this.convOutputElementsPerSample, WORKGROUP_128));
    pass.end();

    this.updateConvUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.convFilterGrad);
    pass.setBindGroup(0, this.bindGroups.convFilterGrad);
    pass.dispatchWorkgroups(ceilDiv(this.convWeightCount, WORKGROUP_64));
    pass.end();

    this.updateReduceConvBiasUniform(batchSize);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.reduce);
    pass.setBindGroup(0, this.bindGroups.reduceConvBias);
    pass.dispatchWorkgroups(ceilDiv(CONV_OUT_CHANNELS, WORKGROUP_64));
    pass.end();

    this.updateScaleUniform(this.uniformBuffers.scaleConvWeights, this.convWeightCount, scaleFactor);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.scale);
    pass.setBindGroup(0, this.bindGroups.scaleGradConvWeights);
    pass.dispatchWorkgroups(ceilDiv(this.convWeightCount, WORKGROUP_128));
    pass.end();

    this.updateScaleUniform(this.uniformBuffers.scaleConvBias, CONV_OUT_CHANNELS, scaleFactor);
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.scale);
    pass.setBindGroup(0, this.bindGroups.scaleGradConvBias);
    pass.dispatchWorkgroups(ceilDiv(CONV_OUT_CHANNELS, WORKGROUP_128));
    pass.end();

    if (this.optimizer === 'sgd') {
      this.updateSgdUniform(this.uniformBuffers.sgdDenseWeights, this.denseWeightCount);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.sgd);
      pass.setBindGroup(0, this.bindGroups.sgdDenseWeights);
      pass.dispatchWorkgroups(ceilDiv(this.denseWeightCount, WORKGROUP_128));
      pass.end();

      this.updateSgdUniform(this.uniformBuffers.sgdDenseBias, this.classes);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.sgd);
      pass.setBindGroup(0, this.bindGroups.sgdDenseBias);
      pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
      pass.end();

      this.updateSgdUniform(this.uniformBuffers.sgdConvWeights, this.convWeightCount);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.sgd);
      pass.setBindGroup(0, this.bindGroups.sgdConvWeights);
      pass.dispatchWorkgroups(ceilDiv(this.convWeightCount, WORKGROUP_128));
      pass.end();

      this.updateSgdUniform(this.uniformBuffers.sgdConvBias, CONV_OUT_CHANNELS);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.sgd);
      pass.setBindGroup(0, this.bindGroups.sgdConvBias);
      pass.dispatchWorkgroups(ceilDiv(CONV_OUT_CHANNELS, WORKGROUP_128));
      pass.end();
    } else {
      this.step += 1;
      this.beta1Power *= this.beta1;
      this.beta2Power *= this.beta2;

      this.updateAdamUniform(this.uniformBuffers.adamDenseWeights, this.denseWeightCount);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.adam);
      pass.setBindGroup(0, this.bindGroups.adamDenseWeights);
      pass.dispatchWorkgroups(ceilDiv(this.denseWeightCount, WORKGROUP_128));
      pass.end();

      this.updateAdamUniform(this.uniformBuffers.adamDenseBias, this.classes);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.adam);
      pass.setBindGroup(0, this.bindGroups.adamDenseBias);
      pass.dispatchWorkgroups(ceilDiv(this.classes, WORKGROUP_128));
      pass.end();

      this.updateAdamUniform(this.uniformBuffers.adamConvWeights, this.convWeightCount);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.adam);
      pass.setBindGroup(0, this.bindGroups.adamConvWeights);
      pass.dispatchWorkgroups(ceilDiv(this.convWeightCount, WORKGROUP_128));
      pass.end();

      this.updateAdamUniform(this.uniformBuffers.adamConvBias, CONV_OUT_CHANNELS);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.adam);
      pass.setBindGroup(0, this.bindGroups.adamConvBias);
      pass.dispatchWorkgroups(ceilDiv(CONV_OUT_CHANNELS, WORKGROUP_128));
      pass.end();
    }
  }

  encodeAccuracyPass(encoder, batchSize) {
    this.updateAccuracyUniform(batchSize);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.accuracy);
    pass.setBindGroup(0, this.bindGroups.accuracy);
    pass.dispatchWorkgroups(ceilDiv(batchSize, WORKGROUP_64));
    pass.end();
  }

  async trainBatch(batchImages, batchLabels, batchSize) {
    if (!this.initialized) {
      throw new Error('Model not initialized');
    }

    this.queue.writeBuffer(this.inputBuffer, 0, batchImages);
    this.queue.writeBuffer(this.labelBuffer, 0, batchLabels);

    const encoder = this.device.createCommandEncoder();
    this.encodeForwardPass(encoder, batchSize);
    this.encodeBackwardPass(encoder, batchSize);
    this.encodeAccuracyPass(encoder, batchSize);
    this.device.queue.submit([encoder.finish()]);

    const losses = await readBufferToArray(this.device, this.lossBuffer, Float32Array, batchSize);
    const accuracyMask = await readBufferToArray(this.device, this.accuracyMaskBuffer, Uint32Array, batchSize);

    let loss = 0;
    for (let i = 0; i < batchSize; i += 1) {
      loss += losses[i];
    }
    loss /= batchSize;

    let correct = 0;
    for (let i = 0; i < batchSize; i += 1) {
      correct += accuracyMask[i];
    }
    const accuracy = correct / batchSize;

    return { loss, accuracy };
  }

  async forwardEvaluate(batchImages, batchLabels, batchSize) {
    this.queue.writeBuffer(this.inputBuffer, 0, batchImages);
    this.queue.writeBuffer(this.labelBuffer, 0, batchLabels);

    const encoder = this.device.createCommandEncoder();
    this.encodeForwardPass(encoder, batchSize);
    this.encodeAccuracyPass(encoder, batchSize);
    this.device.queue.submit([encoder.finish()]);

    const losses = await readBufferToArray(this.device, this.lossBuffer, Float32Array, batchSize);
    const accuracyMask = await readBufferToArray(this.device, this.accuracyMaskBuffer, Uint32Array, batchSize);

    let loss = 0;
    for (let i = 0; i < batchSize; i += 1) {
      loss += losses[i];
    }
    loss /= batchSize;

    let correct = 0;
    for (let i = 0; i < batchSize; i += 1) {
      correct += accuracyMask[i];
    }
    const accuracy = correct / batchSize;

    return { loss, accuracy };
  }

  async evaluateDataset(totalSamples, batchSize, fetchBatch) {
    let totalLoss = 0;
    let totalCorrect = 0;
    let processed = 0;
    const numBatches = Math.ceil(totalSamples / batchSize);
    for (let i = 0; i < numBatches; i += 1) {
      const { images, labels, size } = fetchBatch(batchSize, i);
      const metrics = await this.forwardEvaluate(images, labels, size);
      totalLoss += metrics.loss * size;
      totalCorrect += metrics.accuracy * size;
      processed += size;
    }
    return { loss: totalLoss / processed, accuracy: totalCorrect / processed };
  }

  async predict(vector) {
    const labels = new Float32Array(this.classes);
    await this.forwardEvaluate(vector, labels, 1);
    const probs = await readBufferToArray(this.device, this.probBuffer, Float32Array, this.classes);
    return probs.slice(0, this.classes);
  }

  async export() {
    const [
      convWeights,
      convBias,
      denseWeights,
      denseBias,
      mConvW,
      vConvW,
      mConvB,
      vConvB,
      mDenseW,
      vDenseW,
      mDenseB,
      vDenseB,
    ] = await Promise.all([
      readBufferToArray(this.device, this.convWeightBuffer, Float32Array, this.convWeightCount),
      readBufferToArray(this.device, this.convBiasBuffer, Float32Array, CONV_OUT_CHANNELS),
      readBufferToArray(this.device, this.denseWeightBuffer, Float32Array, this.denseWeightCount),
      readBufferToArray(this.device, this.denseBiasBuffer, Float32Array, this.classes),
      readBufferToArray(this.device, this.mConvWeightBuffer, Float32Array, this.convWeightCount),
      readBufferToArray(this.device, this.vConvWeightBuffer, Float32Array, this.convWeightCount),
      readBufferToArray(this.device, this.mConvBiasBuffer, Float32Array, CONV_OUT_CHANNELS),
      readBufferToArray(this.device, this.vConvBiasBuffer, Float32Array, CONV_OUT_CHANNELS),
      readBufferToArray(this.device, this.mDenseWeightBuffer, Float32Array, this.denseWeightCount),
      readBufferToArray(this.device, this.vDenseWeightBuffer, Float32Array, this.denseWeightCount),
      readBufferToArray(this.device, this.mDenseBiasBuffer, Float32Array, this.classes),
      readBufferToArray(this.device, this.vDenseBiasBuffer, Float32Array, this.classes),
    ]);

    return {
      optimizer: this.optimizer,
      learningRate: this.learningRate,
      beta1Power: this.beta1Power,
      beta2Power: this.beta2Power,
      step: this.step,
      convWeights: Array.from(convWeights),
      convBias: Array.from(convBias),
      denseWeights: Array.from(denseWeights),
      denseBias: Array.from(denseBias),
      mConvWeights: Array.from(mConvW),
      vConvWeights: Array.from(vConvW),
      mConvBias: Array.from(mConvB),
      vConvBias: Array.from(vConvB),
      mDenseWeights: Array.from(mDenseW),
      vDenseWeights: Array.from(vDenseW),
      mDenseBias: Array.from(mDenseB),
      vDenseBias: Array.from(vDenseB),
    };
  }

  async import(state) {
    if (!this.initialized) {
      throw new Error('Model must be initialized before importing');
    }

    if (state.convWeights?.length !== this.convWeightCount || state.convBias?.length !== CONV_OUT_CHANNELS) {
      throw new Error('Invalid convolution parameter dimensions');
    }
    if (state.denseWeights?.length !== this.denseWeightCount || state.denseBias?.length !== this.classes) {
      throw new Error('Invalid dense parameter dimensions');
    }

    writeBuffer(this.device, this.convWeightBuffer, new Float32Array(state.convWeights));
    writeBuffer(this.device, this.convBiasBuffer, new Float32Array(state.convBias));
    writeBuffer(this.device, this.denseWeightBuffer, new Float32Array(state.denseWeights));
    writeBuffer(this.device, this.denseBiasBuffer, new Float32Array(state.denseBias));

    if (state.mConvWeights && state.mConvWeights.length === this.convWeightCount) {
      writeBuffer(this.device, this.mConvWeightBuffer, new Float32Array(state.mConvWeights));
    }
    if (state.vConvWeights && state.vConvWeights.length === this.convWeightCount) {
      writeBuffer(this.device, this.vConvWeightBuffer, new Float32Array(state.vConvWeights));
    }
    if (state.mConvBias && state.mConvBias.length === CONV_OUT_CHANNELS) {
      writeBuffer(this.device, this.mConvBiasBuffer, new Float32Array(state.mConvBias));
    }
    if (state.vConvBias && state.vConvBias.length === CONV_OUT_CHANNELS) {
      writeBuffer(this.device, this.vConvBiasBuffer, new Float32Array(state.vConvBias));
    }
    if (state.mDenseWeights && state.mDenseWeights.length === this.denseWeightCount) {
      writeBuffer(this.device, this.mDenseWeightBuffer, new Float32Array(state.mDenseWeights));
    }
    if (state.vDenseWeights && state.vDenseWeights.length === this.denseWeightCount) {
      writeBuffer(this.device, this.vDenseWeightBuffer, new Float32Array(state.vDenseWeights));
    }
    if (state.mDenseBias && state.mDenseBias.length === this.classes) {
      writeBuffer(this.device, this.mDenseBiasBuffer, new Float32Array(state.mDenseBias));
    }
    if (state.vDenseBias && state.vDenseBias.length === this.classes) {
      writeBuffer(this.device, this.vDenseBiasBuffer, new Float32Array(state.vDenseBias));
    }

    this.optimizer = state.optimizer ?? this.optimizer;
    this.learningRate = state.learningRate ?? this.learningRate;
    this.beta1Power = state.beta1Power ?? this.beta1;
    this.beta2Power = state.beta2Power ?? this.beta2;
    this.step = state.step ?? this.step;
  }

  async getVisualization() {
    const filtersArray = await readBufferToArray(this.device, this.convWeightBuffer, Float32Array, this.convWeightCount);
    const filters = [];
    const filterSize = CONV_KERNEL * CONV_KERNEL * INPUT_CHANNELS;
    for (let i = 0; i < CONV_OUT_CHANNELS; i += 1) {
      filters.push(Array.from(filtersArray.slice(i * filterSize, (i + 1) * filterSize)));
    }

    const activationsArray = await readBufferToArray(this.device, this.poolOutputBuffer, Float32Array, this.poolOutputElementsPerSample);
    const activations = [];
    const mapSize = POOL_WIDTH * POOL_HEIGHT;
    for (let i = 0; i < CONV_OUT_CHANNELS; i += 1) {
      activations.push(Array.from(activationsArray.slice(i * mapSize, (i + 1) * mapSize)));
    }

    return {
      filters,
      activations,
      activationWidth: POOL_WIDTH,
      activationHeight: POOL_HEIGHT,
    };
  }
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
