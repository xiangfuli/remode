// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RMD_SEED_CHECK_CU
#define RMD_SEED_CHECK_CU

#include <mvs_device_data.cuh>
#include <seed_matrix.cuh>
#include "texture_memory.cuh"



namespace rmd
{

#ifndef EXTERN_INCLUDED
#define EXTERN_INCLUDED

extern texture<float, cudaTextureType2D, cudaReadModeElementType> ref_img_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> curr_img_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> mu_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> sigma_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> a_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> b_tex;
extern texture<int, cudaTextureType2D, cudaReadModeElementType> convergence_tex;
extern texture<float2, cudaTextureType2D, cudaReadModeElementType> epipolar_matches_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> g_tex;
// Pre-computed template statistics
extern texture<float, cudaTextureType2D, cudaReadModeElementType> sum_templ_tex;
extern texture<float, cudaTextureType2D, cudaReadModeElementType> const_templ_denom_tex;

#endif

__global__
void seedCheckKernel(mvs::DeviceData *dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  if(x > dev_ptr->width-RMD_CORR_PATCH_SIDE-1 || y > dev_ptr->height-RMD_CORR_PATCH_SIDE-1 ||
     x < RMD_CORR_PATCH_SIDE || y < RMD_CORR_PATCH_SIDE)
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::BORDER;
    return;
  }

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  // Retrieve current estimations of parameters
  const float mu = tex2D(mu_tex, xx, yy);
  const float sigma_sq = tex2D(sigma_tex, xx, yy);
  const float a = tex2D(a_tex, xx, yy);
  const float b = tex2D(b_tex, xx, yy);

  // if E(inlier_ratio) > eta_inlier && sigma_sq < epsilon
  if( ((a / (a + b)) > dev_ptr->eta_inlier)
      && (sigma_sq < dev_ptr->epsilon) )
  { // The seed converged
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::CONVERGED;
  }
  else if((a-1) / (a + b - 2) < dev_ptr->eta_outlier)
  { // The seed failed to converge
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::DIVERGED;
  }
  else
  {
    dev_ptr->convergence->atXY(x, y) = ConvergenceStates::UPDATE;
  }
}

} // rmd namespace

#endif
