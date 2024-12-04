#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx) {
    // 检查输入张量的维度
    TORCH_CHECK(ref.dim() == 3, "ref张量必须是3维的");
    TORCH_CHECK(query.dim() == 3, "query张量必须是3维的");
    TORCH_CHECK(idx.dim() == 3, "idx张量必须是3维的");
    TORCH_CHECK(ref.size(0) == query.size(0), "ref和query的batch维度必须相等");
    TORCH_CHECK(ref.size(1) == query.size(1), "ref和query的特征维度必须相等");

    // 获取张量的维度信息
    auto batch = ref.size(0);
    auto dim = ref.size(1);
    auto ref_nb = ref.size(2);
    auto query_nb = query.size(2);
    auto k = idx.size(1);

    // 获取指针
    auto* ref_dev = ref.data_ptr<float>();
    auto* query_dev = query.data_ptr<float>();
    auto* idx_dev = idx.data_ptr<long>();


if (ref.is_cuda()) {
#ifdef WITH_CUDA
    // 使用CUDA设备分配内存
    float* dist_dev = nullptr;
    cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void**>(&dist_dev), ref_nb * query_nb * sizeof(float));
    TORCH_CHECK(alloc_err == cudaSuccess, "CUDA内存分配失败: ", cudaGetErrorString(alloc_err));

    // 循环处理每个batch
    for (int b = 0; b < batch; b++) {
        knn_device(
            ref_dev + b * dim * ref_nb, ref_nb,
            query_dev + b * dim * query_nb, query_nb,
            dim, k, dist_dev,
            idx_dev + b * k * query_nb,
            c10::cuda::getCurrentCUDAStream()
        );
    }

    // 释放分配的内存
    cudaFree(dist_dev);

    // 检查CUDA调用是否发生错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "KNN计算时发生CUDA错误: ", cudaGetErrorString(err));

    return 1;
#else
    AT_ERROR("未编译为支持GPU的版本");
#endif
}


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
