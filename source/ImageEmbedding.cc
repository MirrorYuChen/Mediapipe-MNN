/*
 * @Author: chenjingyu
 * @Date: 2023-08-04 20:30:02
 * @LastEditTime: 2023-08-16 10:33:59
 * @Description: Image Embedding
 * @FilePath: \Mediapipe-MNN\source\ImageEmbedding.cc
 */
#include "ImageEmbedding.h"
#include <iostream>
#include <cmath>
#include "Utils.h"
#include "opencv2/opencv.hpp"

namespace mirror {
using namespace MNN;
ImageEmbedding::~ImageEmbedding() {
  net_->releaseModel();
  net_->releaseSession(sess_);  
}

bool ImageEmbedding::LoadModel(const char *model_file) {
  std::cout << "Start load model." << std::endl;
  // 1.load model
  net_ = std::unique_ptr<Interpreter>(Interpreter::createFromFile(model_file));
  if (net_ == nullptr) {
    std::cout << "Failed load model." << std::endl;
    return false;
  }

  // 2.create session
  ScheduleConfig schedule_config;
  schedule_config.type = MNN_FORWARD_CPU;
  schedule_config.numThread = 4;
  sess_ = net_->createSession(schedule_config);
  input_tensor_ = net_->getSessionInput(sess_, nullptr);

  input_h_ = input_tensor_->height();
  input_w_ = input_tensor_->width();

  std::cout << "End load model." << std::endl;
  inited_ = true;
  return true;
}

void ImageEmbedding::setFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::RGB;
  image_process_config.wrap = CV::ZERO;
  memcpy(image_process_config.mean, meanVals_, sizeof(meanVals_));
  memcpy(image_process_config.normal, normVals_, sizeof(normVals_));
  pretreat_ = std::shared_ptr<CV::ImageProcess>(
      CV::ImageProcess::create(image_process_config));
}

bool ImageEmbedding::Detect(const ImageHead &in, RotateType type, Embedding &embedding) {
  std::cout << "Start detect." << std::endl;
  embedding.float_embedding.clear();
  embedding.quantized_embedding.clear();
  if (in.data == nullptr) {
    std::cout << "Error Input empty." << std::endl;
    return false;
  }

  if (!inited_) {
    std::cout << "Model Uninitialized." << std::endl;
    return false;
  }

  // 1.set input
  int width = in.width;
  int height = in.height;
  std::vector<Point2f> input_region = getInputRegion(in, type, input_w_, input_h_, false);
  float points_src[] = {
    input_region[0].x, input_region[0].y, 
    input_region[1].x, input_region[1].y,
    input_region[2].x, input_region[2].y,
    input_region[3].x, input_region[3].y,
  };
  float points_dst[] = {
    0.0f,  0.0f,
    0.0f,  (float)(input_h_ - 1),
    (float)(input_w_ - 1), 0.0f,
    (float)(input_w_ - 1), (float)(input_h_ - 1),
  };
  trans_.setPolyToPoly((CV::Point *)points_dst, (CV::Point *)points_src, 4);
  pretreat_->setMatrix(trans_);
  pretreat_->convert((uint8_t *)in.data, width, height, in.width_step, input_tensor_);

  // 2.do inference
  int ret = net_->runSession(sess_);
  if (ret != 0) {
    std::cout << "Failed do inference." << std::endl;
    return -1;
  }

  // 3.get the result
  MNN::Tensor *result = net_->getSessionOutput(sess_, nullptr);
  if (nullptr == result) {
    std::cout << "Error output." << std::endl;
    return false;
  }
  std::shared_ptr<MNN::Tensor> output_result(new MNN::Tensor(result, result->getDimensionType()));
  result->copyToHostTensor(output_result.get());

  int channel = output_result->channel();
  embedding = FillQuantizedEmbedding(output_result->host<float>(), channel, true);  

  std::cout << "End detect." << std::endl;
  return true;
}

} // namespace mirror
