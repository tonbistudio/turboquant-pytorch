#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_GROUPS 128
#define MAX_LEVELS 256

template <typename T>
__device__ float convert_to_float(T value) { return 0.0f; }
template <> __device__ float convert_to_float<c10::Half>(c10::Half value) { return __half2float(value); }
template <> __device__ float convert_to_float<float>(float value) { return value; }
template <> __device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) { return static_cast<float>(value); }

template <typename T>
__device__ T convert_from_float(float value) { return static_cast<T>(0); }
template <> __device__ c10::Half convert_from_float<c10::Half>(float value) { return __float2half(value); }
template <> __device__ float convert_from_float<float>(float value) { return value; }
template <> __device__ at::BFloat16 convert_from_float<at::BFloat16>(float value) { return static_cast<at::BFloat16>(value); }

/*
 * Sparse geometric product: rotor * multivector
 *
 * Rotor R in Cl(3,0) has only 4 non-zero components: [s, 0, 0, 0, b12, b13, b23, 0]
 * This eliminates ~50% of FMAs vs the full 8x8 product.
 *
 * Full Cl(3,0) multiplication table with e_i*e_i = +1:
 *   r0 = s*b0 - b12*x12 - b13*x13 - b23*x23
 *   r1 = s*b1 + b12*x2 + b13*x3 + b23*x123
 *   r2 = s*b2 - b12*x1 + b23*x3 - b13*x123
 *   r3 = s*b3 - b13*x1 - b23*x2 + b12*x123
 *   r4 = s*x12 + b12*x0
 *   r5 = s*x13 + b13*x0
 *   r6 = s*x23 + b23*x0
 *   r7 = s*x123 - b23*x1 + b13*x2 - b12*x3
 */
__device__ void gp_rotor_mv(
    float s, float p12, float p13, float p23,
    const float x[8], float r[8])
{
    r[0] = s*x[0] - p12*x[4] - p13*x[5] - p23*x[6];
    r[1] = s*x[1] + p12*x[2] + p13*x[3] + p23*x[7];
    r[2] = s*x[2] - p12*x[1] + p23*x[3] - p13*x[7];
    r[3] = s*x[3] - p13*x[1] - p23*x[2] + p12*x[7];
    r[4] = s*x[4] + p12*x[0];
    r[5] = s*x[5] + p13*x[0];
    r[6] = s*x[6] + p23*x[0];
    r[7] = s*x[7] - p23*x[1] + p13*x[2] - p12*x[3];
}

__device__ float quantize_scalar(float val, const float* __restrict__ centroids, int levels)
{
    float best = centroids[0];
    float min_d = fabsf(val - best);
    #pragma unroll
    for (int i = 1; i < levels; ++i) {
        float d = fabsf(val - centroids[i]);
        if (d < min_d) { min_d = d; best = centroids[i]; }
    }
    return best;
}

/*
 * Fused RotorQuant kernel:
 *   embed → rotor_sandwich_fwd → quantize → rotor_sandwich_inv → extract
 *
 * One block per batch item. Threads iterate over groups.
 * Rotors and centroids loaded into shared memory.
 */
template <typename T>
__global__ void rotor_full_fused_kernel(
    const T* __restrict__ input,     // (batch, emb_dim)
    const float* __restrict__ rotors, // (n_groups, 4): [s, b12, b13, b23]
    const float* __restrict__ c_scalar,   int n_scalar,
    const float* __restrict__ c_vector,   int n_vector,
    const float* __restrict__ c_bivector, int n_bivector,
    const float* __restrict__ c_trivector,int n_trivector,
    T* __restrict__ output,          // (batch, emb_dim)
    int batch_size, int emb_dim, int n_groups)
{
    // Shared memory for rotors and centroids
    __shared__ float sh_rotors[MAX_GROUPS * 4];
    __shared__ float sh_c_scalar[MAX_LEVELS];
    __shared__ float sh_c_vector[MAX_LEVELS];
    __shared__ float sh_c_bivector[MAX_LEVELS];
    __shared__ float sh_c_trivector[MAX_LEVELS];

    int tid = threadIdx.x;

    // Cooperatively load rotors
    for (int i = tid; i < n_groups * 4; i += blockDim.x)
        sh_rotors[i] = rotors[i];

    // Load centroids
    for (int i = tid; i < n_scalar; i += blockDim.x)
        sh_c_scalar[i] = c_scalar[i];
    for (int i = tid; i < n_vector; i += blockDim.x)
        sh_c_vector[i] = c_vector[i];
    for (int i = tid; i < n_bivector; i += blockDim.x)
        sh_c_bivector[i] = c_bivector[i];
    for (int i = tid; i < n_trivector; i += blockDim.x)
        sh_c_trivector[i] = c_trivector[i];

    __syncthreads();

    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input + b * emb_dim;
    T* out_ptr = output + b * emb_dim;

    // Each thread handles one or more groups
    for (int g = tid; g < n_groups; g += blockDim.x) {
        // Load rotor
        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        // Embed: 3 vector dims → multivector (grade-1 only)
        int d0 = g * 3;
        float x_mv[8] = {0.0f};
        if (d0     < emb_dim) x_mv[1] = convert_to_float(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float(in_ptr[d0 + 2]);

        // Forward sandwich: temp = R * x, rotated = temp * R̃
        float temp[8], rotated[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
        gp_rotor_mv(s, -p12, -p13, -p23, temp, rotated);

        // Grade-aware quantization
        float q_mv[8];
        q_mv[0] = quantize_scalar(rotated[0], sh_c_scalar,   n_scalar);
        q_mv[1] = quantize_scalar(rotated[1], sh_c_vector,   n_vector);
        q_mv[2] = quantize_scalar(rotated[2], sh_c_vector,   n_vector);
        q_mv[3] = quantize_scalar(rotated[3], sh_c_vector,   n_vector);
        q_mv[4] = quantize_scalar(rotated[4], sh_c_bivector, n_bivector);
        q_mv[5] = quantize_scalar(rotated[5], sh_c_bivector, n_bivector);
        q_mv[6] = quantize_scalar(rotated[6], sh_c_bivector, n_bivector);
        q_mv[7] = quantize_scalar(rotated[7], sh_c_trivector,n_trivector);

        // Inverse sandwich: temp' = R̃ * q, final = temp' * R
        float temp2[8], final_mv[8];
        gp_rotor_mv(s, -p12, -p13, -p23, q_mv, temp2);
        gp_rotor_mv(s, p12, p13, p23, temp2, final_mv);

        // Extract vector grades back to output
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(final_mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(final_mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(final_mv[3]);
    }
}

/*
 * Standalone rotor sandwich (no quantization).
 * Useful for the inner_product path where we only need rotor decorrelation.
 */
template <typename T>
__global__ void rotor_sandwich_kernel(
    const T* __restrict__ input,      // (batch, emb_dim)
    const float* __restrict__ rotors, // (n_groups, 4)
    T* __restrict__ output,           // (batch, n_groups, 8)
    int batch_size, int emb_dim, int n_groups)
{
    __shared__ float sh_rotors[MAX_GROUPS * 4];

    int tid = threadIdx.x;
    for (int i = tid; i < n_groups * 4; i += blockDim.x)
        sh_rotors[i] = rotors[i];
    __syncthreads();

    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input + b * emb_dim;
    T* out_ptr = output + b * n_groups * 8;

    for (int g = tid; g < n_groups; g += blockDim.x) {
        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        int d0 = g * 3;
        float x_mv[8] = {0.0f};
        if (d0     < emb_dim) x_mv[1] = convert_to_float(in_ptr[d0]);
        if (d0 + 1 < emb_dim) x_mv[2] = convert_to_float(in_ptr[d0 + 1]);
        if (d0 + 2 < emb_dim) x_mv[3] = convert_to_float(in_ptr[d0 + 2]);

        float temp[8], rotated[8];
        gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
        gp_rotor_mv(s, -p12, -p13, -p23, temp, rotated);

        int base = g * 8;
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            out_ptr[base + c] = convert_from_float<T>(rotated[c]);
    }
}

/*
 * Inverse rotor sandwich: reconstruct vectors from multivectors.
 */
template <typename T>
__global__ void rotor_inverse_sandwich_kernel(
    const T* __restrict__ input_mv,   // (batch, n_groups, 8)
    const float* __restrict__ rotors, // (n_groups, 4)
    T* __restrict__ output,           // (batch, emb_dim)
    int batch_size, int emb_dim, int n_groups)
{
    __shared__ float sh_rotors[MAX_GROUPS * 4];

    int tid = threadIdx.x;
    for (int i = tid; i < n_groups * 4; i += blockDim.x)
        sh_rotors[i] = rotors[i];
    __syncthreads();

    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input_mv + b * n_groups * 8;
    T* out_ptr = output + b * emb_dim;

    for (int g = tid; g < n_groups; g += blockDim.x) {
        float s   = sh_rotors[g * 4 + 0];
        float p12 = sh_rotors[g * 4 + 1];
        float p13 = sh_rotors[g * 4 + 2];
        float p23 = sh_rotors[g * 4 + 3];

        int base = g * 8;
        float q_mv[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c)
            q_mv[c] = convert_to_float(in_ptr[base + c]);

        float temp[8], final_mv[8];
        gp_rotor_mv(s, -p12, -p13, -p23, q_mv, temp);
        gp_rotor_mv(s, p12, p13, p23, temp, final_mv);

        int d0 = g * 3;
        if (d0     < emb_dim) out_ptr[d0]     = convert_from_float<T>(final_mv[1]);
        if (d0 + 1 < emb_dim) out_ptr[d0 + 1] = convert_from_float<T>(final_mv[2]);
        if (d0 + 2 < emb_dim) out_ptr[d0 + 2] = convert_from_float<T>(final_mv[3]);
    }
}

// ─── Template instantiations and pybind11 ───

template <typename T>
torch::Tensor rotor_full_fused_impl(
    torch::Tensor input, torch::Tensor rotors,
    torch::Tensor c_scalar, int n_scalar,
    torch::Tensor c_vector, int n_vector,
    torch::Tensor c_bivector, int n_bivector,
    torch::Tensor c_trivector, int n_trivector)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 2) / 3;

    auto output = torch::empty_like(input);

    int threads = min(256, max(n_groups, WARP_SIZE));
    dim3 blocks(batch_size);

    rotor_full_fused_kernel<T><<<blocks, threads>>>(
        input.data_ptr<T>(),
        rotors.data_ptr<float>(),
        c_scalar.data_ptr<float>(), n_scalar,
        c_vector.data_ptr<float>(), n_vector,
        c_bivector.data_ptr<float>(), n_bivector,
        c_trivector.data_ptr<float>(), n_trivector,
        output.data_ptr<T>(),
        batch_size, emb_dim, n_groups);

    return output;
}

template <typename T>
torch::Tensor rotor_sandwich_impl(
    torch::Tensor input, torch::Tensor rotors)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 2) / 3;

    auto output = torch::empty({batch_size, n_groups, 8}, input.options());

    int threads = min(256, max(n_groups, WARP_SIZE));
    rotor_sandwich_kernel<T><<<batch_size, threads>>>(
        input.data_ptr<T>(),
        rotors.data_ptr<float>(),
        output.data_ptr<T>(),
        batch_size, emb_dim, n_groups);

    return output;
}

template <typename T>
torch::Tensor rotor_inverse_impl(
    torch::Tensor input_mv, torch::Tensor rotors, int emb_dim)
{
    int batch_size = input_mv.size(0);
    int n_groups = input_mv.size(1);

    auto output = torch::empty({batch_size, emb_dim}, input_mv.options());

    int threads = min(256, max(n_groups, WARP_SIZE));
    rotor_inverse_sandwich_kernel<T><<<batch_size, threads>>>(
        input_mv.data_ptr<T>(),
        rotors.data_ptr<float>(),
        output.data_ptr<T>(),
        batch_size, emb_dim, n_groups);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Fused full pipeline
    m.def("rotor_full_fused_float", &rotor_full_fused_impl<float>);
    m.def("rotor_full_fused_half", &rotor_full_fused_impl<c10::Half>);
    m.def("rotor_full_fused_bf16", &rotor_full_fused_impl<at::BFloat16>);

    // Standalone sandwich (forward)
    m.def("rotor_sandwich_float", &rotor_sandwich_impl<float>);
    m.def("rotor_sandwich_half", &rotor_sandwich_impl<c10::Half>);
    m.def("rotor_sandwich_bf16", &rotor_sandwich_impl<at::BFloat16>);

    // Inverse sandwich
    m.def("rotor_inverse_float", &rotor_inverse_impl<float>);
    m.def("rotor_inverse_half", &rotor_inverse_impl<c10::Half>);
    m.def("rotor_inverse_bf16", &rotor_inverse_impl<at::BFloat16>);
}
