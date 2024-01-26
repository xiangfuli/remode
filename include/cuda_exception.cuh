#include <sstream>
#include <cuda_runtime.h>

namespace rmd
{

struct CudaException : public std::exception
{
  CudaException(const std::string& what, cudaError err)
    : err_(err) {
        std::stringstream description;
        description << "CudaException: " << what << ". ";
        if(err_ != cudaSuccess)
        {
            description << "cudaError code: " << cudaGetErrorString(err_);
            description << " (" << err_ << ")";
        }
        this->what_ = description.str();
  }
  virtual ~CudaException() throw() {}
  const char* what() const noexcept override
  {
    return what_.c_str();
  }
  std::string what_;
  cudaError err_;
};

} // namespace rmd