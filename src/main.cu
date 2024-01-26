#include "cuda_runtime.h"
#include "cuda_texture_types.h"

namespace rmd {
    texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
    texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;

    texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
    texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
    texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
    texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;

    texture<int, cudaTextureType2D, cudaReadModeElementType> convergence_tex;
    texture<float2, cudaTextureType2D, cudaReadModeElementType> epipolar_matches_tex;

    texture<float, cudaTextureType2D, cudaReadModeElementType> g_tex;

    // Pre-computed template statistics
    texture<float, cudaTextureType2D, cudaReadModeElementType> sum_templ_tex;
    texture<float, cudaTextureType2D, cudaReadModeElementType> const_templ_denom_tex;
}

#include "depthmap.h"
#include <thread>
#include <chrono>
int main() {
    rmd::Depthmap depth_map(1000000, 200, 400, 400, 500, 100);

    while(1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}