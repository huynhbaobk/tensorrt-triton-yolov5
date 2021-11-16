#pragma once

#include "NvInfer.h"
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <algorithm>
#include "./cuda_utils.h"
#include "common.hpp"

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
    Int8EntropyCalibrator2(int batchsize, int input_w, int input_h,
    const char* img_dir, const char* calib_table_name,
    const char* input_blob_name, bool read_cache = true);

    virtual ~Int8EntropyCalibrator2();
    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

 private:
    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    const char* input_blob_name_;
    bool read_cache_;
    void* device_input_;
    std::vector<char> calib_cache_;
};

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize,
int input_w, int input_h, const char* img_dir,
const char* calib_table_name, const char* input_blob_name,
bool read_cache)
    : batchsize_(batchsize)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache) {
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const {
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) {
    if (img_idx_ + batchsize_ > static_cast<int>(img_files_.size())) {
        return false;
    }

    std::vector<float> input_imgs_(input_count_, 0);
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        temp = preprocessImg(temp, input_w_, input_h_);
        if (temp.empty()) {
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        for (int ind = 0; ind < input_w_*input_h_*3; ind++)
            input_imgs_[(i-img_idx_)*input_w_*input_h_*3 + ind] = static_cast<float>(*(temp.data + ind));
    }
    img_idx_ += batchsize_;

    CUDA_CHECK(cudaMemcpy(device_input_, input_imgs_.data(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) {
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) {
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
