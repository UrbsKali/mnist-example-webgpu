@group(0) @binding(0) var<storage, read> inputTensor : array<f32>;
@group(0) @binding(1) var<storage, read_write> outputTensor : array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&outputTensor)) {
    return;
  }
  let value = inputTensor[idx];
  outputTensor[idx] = max(0.0, value);
}
