/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/LocalSearchQuantizer.h>
#include <memory>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;
struct IcmEncoderImpl;

class GpuIcmEncoder : public lsq::IcmEncoder {
   public:
    GpuIcmEncoder(size_t M, size_t K, GpuResourcesProvider* prov);

    void set_unary_term(size_t n, const float* unaries) override;

    void set_binary_term(const float* binaries) override;

    void encode(
            const float* x,
            const float* codebooks,
            int32_t* codes,
            std::mt19937& gen,
            size_t n,
            size_t d,
            size_t nperts,
            size_t ils_iters,
            size_t icm_iters) const override;

   private:
    IcmEncoderImpl* encoder;
};

struct GpuIcmEncoderFactory : public lsq::IcmEncoderFactory {
    GpuIcmEncoderFactory();

    lsq::IcmEncoder* get(size_t M, size_t K) override;

    GpuResourcesProvider* prov;
};

} // namespace gpu
} // namespace faiss
