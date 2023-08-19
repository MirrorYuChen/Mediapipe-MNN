/*
 * @Author: chenjingyu
 * @Date: 2023-07-30 20:40:28
 * @LastEditTime: 2023-08-19 22:33:09
 * @Description: face detector lite
 * @FilePath: \Mediapipe-MNN\source\FaceDetectorLite.cc
 */
#include "FaceDetectorLite.h"

#include "Utils.h"
#include <iostream>

using namespace MNN;
namespace mirror {
static constexpr float BIASES[] = {1.0f, 1.0f, 3.0f, 3.0f,  5.0f,
                                   5.0f, 7.0f, 7.0f, 10.0f, 10.0f};

FaceDetectorLite::~FaceDetectorLite() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool FaceDetectorLite::LoadModel(const char *model_file) {
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

void FaceDetectorLite::setFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::BGR;
  image_process_config.wrap = CV::ZERO;
  memcpy(image_process_config.mean, meanVals_, sizeof(meanVals_));
  memcpy(image_process_config.normal, normVals_, sizeof(normVals_));
  pretreat_ = std::shared_ptr<CV::ImageProcess>(
      CV::ImageProcess::create(image_process_config));
}

bool FaceDetectorLite::Detect(const ImageHead &in, RotateType type,
                              std::vector<ObjectInfo> &objects) {
  std::cout << "Start detect." << std::endl;
  objects.clear();
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
  // clang-format off
  std::vector<Point2f> input_region = getInputRegion(in, type, input_w_, input_h_);
  float points_src[] = {
    input_region[0].x, input_region[0].y,
    input_region[1].x, input_region[1].y,
    input_region[2].x, input_region[2].y,
    input_region[3].x, input_region[3].y,
  };
  float points_dst[] = {
    0.0f, 0.0f,
    0.0f, (float)(input_h_ - 1),
    (float)(input_w_ - 1), 0.0f,
    (float)(input_w_ - 1), (float)(input_h_ - 1),
  };
  // clang-format on
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
  MNN::Tensor *result = net_->getSessionOutput(sess_, "conv_finalout");
  if (nullptr == result) {
    std::cout << "Error output." << std::endl;
    return false;
  }
  std::shared_ptr<MNN::Tensor> output_result(
      new MNN::Tensor(result, result->getDimensionType()));
  result->copyToHostTensor(output_result.get());

  // 4.parse the result
  float *result_ptr = output_result->host<float>();
  int rows = output_result->height();
  int cols = output_result->width();
  int channel = output_result->channel();
  int channel_step = cols * rows;

  ObjectInfo object;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int index = 5 * channel_step;
      for (int j = 0; j < boxNum_; ++j) {
        int index_boxes = index * j + row * cols + col;
        int index_score = index_boxes + 4 * channel_step;

        float score = sigmoid(result_ptr[index_score]);
        if (score < score_thresh_) continue;

        float cx = (col + sigmoid(result_ptr[index_boxes + 0 * channel_step])) / cols;
        float cy = (row + sigmoid(result_ptr[index_boxes + 1 * channel_step])) / rows;
        float w = exp(result_ptr[index_boxes + 2 * channel_step]) * BIASES[2 * j] / cols;
        float h = exp(result_ptr[index_boxes + 3 * channel_step]) * BIASES[2 * j + 1] / rows;

        Point2f tl, br, tl_origin, br_origin;
        tl.x = (cx - 0.5f * w) * input_w_;
        tl.y = (cy - 0.5f * h) * input_h_;
        br.x = (cx + 0.5f * w) * input_w_;
        br.y = (cy + 0.5f * h) * input_h_;

        tl_origin.x = trans_[0] * tl.x + trans_[1] * tl.y + trans_[2];
        tl_origin.y = trans_[3] * tl.x + trans_[4] * tl.y + trans_[5];
        br_origin.x = trans_[0] * br.x + trans_[1] * br.y + trans_[2];
        br_origin.y = trans_[3] * br.x + trans_[4] * br.y + trans_[5];

        object.rect.left = MIN_(tl_origin.x, br_origin.x);
        object.rect.top = MIN_(tl_origin.y, br_origin.y);
        object.rect.right = MAX_(tl_origin.x, br_origin.x);
        object.rect.bottom = MAX_(tl_origin.y, br_origin.y);

        objects.emplace_back(object);
      }
    }
  }
  NMSObjects(objects, iou_thresh_);

  std::cout << "End detect." << std::endl;
  return true;
}

} // namespace mirror
