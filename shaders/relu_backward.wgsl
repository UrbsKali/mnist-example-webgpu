@group(0) @binding(0) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(1) var<storage, read> activations : array<f32>;
@group(0) @binding(2) var<storage, read_write> gradInput : array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&gradInput)) {
    return;
  }
  let grad = gradOutput[idx];
  let act = activations[idx];
  gradInput[idx] = select(0.0, grad, act > 0.0);
}
