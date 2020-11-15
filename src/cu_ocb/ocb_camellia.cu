#include "cu_ocb/ocb_camellia.h"
#include "cu_ocb/ocb_camellia_impl.cuh"

namespace cu_ocb
{
OcbCamellia::OcbCamellia(OcbConfig config)
    : impl_{new OcbCamelliaImpl(std::move(config))}
{
}

OcbCamellia::~OcbCamellia() { delete impl_; }

void OcbCamellia::generateKeytable(std::string_view key)
{
  try
    {
      impl_->generateKeytable(key);
    }
  catch (std::exception& err)
    {
      std::cerr << "error in " << __func__ << ": " << err.what() << std::endl;
    }
}

void OcbCamellia::setKeytable(void* keytable)
{
  try
    {
      impl_->setKeytable(keytable);
    }
  catch (std::exception& err)
    {
      std::cerr << "error in " << __func__ << ": " << err.what() << std::endl;
    }
}

size_t OcbCamellia::encrypt(std::string_view data, size_t index,
                            std::string_view L, Block& check_sum,
                            Block& last_offset, void* result, bool encrypt)
{
  try
    {
      return impl_->encrypt(data, index, L, check_sum, last_offset, result,
                            encrypt);
    }
  catch (std::exception& err)
    {
      std::cerr << "error in " << __func__ << ": " << err.what() << std::endl;
      return 0;
    }
}

const GpuTimeSpent* OcbCamellia::gpuTimeSpent() const
{
  try
    {
      return impl_->gpuTimeSpent();
    }
  catch (std::exception& err)
    {
      std::cerr << "error in " << __func__ << ": " << err.what() << std::endl;
      return nullptr;
    }
}

bool OcbCamellia::hasError() const { return impl_->hasError(); }

}  // namespace cu_ocb
