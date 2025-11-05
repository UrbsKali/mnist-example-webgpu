const shaderCache = new Map();

export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('Failed to acquire GPU adapter');
  }

  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
  });

  const queue = device.queue;
  return { adapter, device, queue };
}

export async function loadShaderModule(device, url) {
  if (shaderCache.has(url)) {
    return shaderCache.get(url);
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load shader ${url}`);
  }
  const code = await response.text();
  const module = device.createShaderModule({ code, label: url });
  shaderCache.set(url, module);
  return module;
}

function getArrayView(arrayLike) {
  if (arrayLike instanceof ArrayBuffer) {
    return new Uint8Array(arrayLike);
  }
  if (ArrayBuffer.isView(arrayLike)) {
    return new Uint8Array(arrayLike.buffer, arrayLike.byteOffset, arrayLike.byteLength);
  }
  throw new Error('Unsupported buffer data type');
}

export function createBuffer(device, array, usage, label) {
  const byteLength = array.byteLength ?? getArrayView(array).byteLength;
  const buffer = device.createBuffer({
    label,
    size: align(byteLength, 4),
    usage,
    mappedAtCreation: true,
  });
  const view = getArrayView(array);
  new Uint8Array(buffer.getMappedRange()).set(view);
  buffer.unmap();
  return buffer;
}

export function createEmptyBuffer(device, byteLength, usage, label) {
  return device.createBuffer({
    label,
    size: align(byteLength, 4),
    usage,
  });
}

export function writeBuffer(device, buffer, data, offset = 0) {
  const view = getArrayView(data);
  device.queue.writeBuffer(buffer, offset, view, 0, view.byteLength);
}

export async function readBufferToArray(device, buffer, constructor, length) {
  const readBuffer = device.createBuffer({
    size: align(constructor.BYTES_PER_ELEMENT * length, 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, readBuffer.size);
  device.queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const copyArray = new constructor(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  readBuffer.destroy();
  return copyArray;
}

export function createComputePipeline(device, module, entryPoint, bindGroupLayouts, label) {
  return device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts }),
    compute: {
      module,
      entryPoint,
    },
  });
}

export function createBindGroup(device, layout, entries, label) {
  return device.createBindGroup({
    label,
    layout,
    entries,
  });
}

export function align(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

export function ceilDiv(a, b) {
  return Math.floor((a + b - 1) / b);
}

export function createTimer(device) {
  if (!device.features.has('timestamp-query')) {
    return {
      enabled: false,
      async measure(callback) {
        const start = performance.now();
        await callback();
        const end = performance.now();
        return end - start;
      },
    };
  }

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolveBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    enabled: true,
    async measure(callback) {
      const encoder = device.createCommandEncoder();
      encoder.writeTimestamp(querySet, 0);
      await callback(encoder);
      encoder.writeTimestamp(querySet, 1);
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      device.queue.submit([encoder.finish()]);
      await resolveBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(resolveBuffer.getMappedRange());
      const elapsed = Number((timestamps[1] - timestamps[0]) * BigInt(device.limits.timestampPeriod ?? 1)) / 1_000_000;
      resolveBuffer.unmap();
      return elapsed;
    },
  };
}
