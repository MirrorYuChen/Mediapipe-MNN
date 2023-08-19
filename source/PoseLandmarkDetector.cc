/*
 * @Author: chenjingyu
 * @Date: 2023-08-20 00:14:23
 * @LastEditTime: 2023-08-20 01:46:08
 * @Description: Pose Landmark Detector
 * @FilePath: \Mediapipe-MNN\source\PoseLandmarkDetector.cc
 */
#include "PoseLandmarkDetector.h"
#include "Utils.h"
#include <cfloat>
#include <iostream>

using namespace MNN;
namespace mirror {
PoseLandmarkDetector::~PoseLandmarkDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool PoseLandmarkDetector::LoadModel(const char *model_file) {
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

void PoseLandmarkDetector::setFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::RGB;
  image_process_config.wrap = CV::ZERO;
  memcpy(image_process_config.mean, meanVals_, sizeof(meanVals_));
  memcpy(image_process_config.normal, normVals_, sizeof(normVals_));
  pretreat_ = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(image_process_config));
}

bool PoseLandmarkDetector::Detect(const ImageHead &in, RotateType type,
                                std::vector<ObjectInfo> &objects) {
  std::cout << "Start detect." << std::endl;
  if (in.data == nullptr) {
    std::cout << "Error Input empty." << std::endl;
    return false;
  }

  if (!inited_) {
    std::cout << "Model Uninitialized." << std::endl;
    return false;
  }
  int width = in.width;
  int height = in.height;
  for (auto &object : objects) {
    std::vector<Point2f> region = getInputRegion(in, type, object, scale_);
    float points_src[] = {
      region[0].x, region[0].y, region[1].x, region[1].y,
      region[2].x, region[2].y, region[3].x, region[3].y,
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


    // 1.5.do inference
    int ret = net_->runSession(sess_);
    if (ret != 0) {
      std::cout << "Failed do inference." << std::endl;
      return -1;
    }

    // 1.6 get the result
    MNN::Tensor *landmark = net_->getSessionOutput(sess_, "Identity");
    MNN::Tensor *score = net_->getSessionOutput(sess_, "Identity_1");
    MNN::Tensor *mask = net_->getSessionOutput(sess_, "Identity_2");
    MNN::Tensor *headmap = net_->getSessionOutput(sess_, "Identity_3");
    MNN::Tensor *landmark_world = net_->getSessionOutput(sess_, "Identity_4");

    std::shared_ptr<MNN::Tensor> output_landmark(
        new MNN::Tensor(landmark, landmark->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_score(
        new MNN::Tensor(score, score->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_mask(
        new MNN::Tensor(mask, mask->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_headmap(
        new MNN::Tensor(headmap, headmap->getDimensionType()));    
    std::shared_ptr<MNN::Tensor> output_landmark_world(
        new MNN::Tensor(landmark_world, landmark_world->getDimensionType()));

    landmark->copyToHostTensor(output_landmark.get());
    score->copyToHostTensor(output_score.get());
    mask->copyToHostTensor(output_mask.get());
    headmap->copyToHostTensor(output_headmap.get());
    landmark_world->copyToHostTensor(output_landmark_world.get());

    float *score_ptr = output_score->host<float>();
    if (score_ptr[0] < 0.8f) continue;

    object.landmarks3d.resize(kNumPoseLandmarks);
    float *landmarks_ptr = output_landmark->host<float>();
    for (int k = 0; k < kNumPoseLandmarks; ++k) {
      object.landmarks3d[k].x  = landmarks_ptr[5 * k + 0];
      object.landmarks3d[k].y = landmarks_ptr[5 * k + 1];
      object.landmarks3d[k].z = landmarks_ptr[5 * k + 2];
      object.landmarks3d[k].visibility = landmarks_ptr[5 * k + 3];
      object.landmarks3d[k].presence = sigmoid(landmarks_ptr[5 * k + 4]);
    }

    // MNN only change the data stored format: NCHW
    // but the parse process should also according to NHWC
    float *headmap_ptr = output_headmap->host<float>();
    int hm_width = 64;
    int hm_height = 64;
    int hm_channels = kNumPoseLandmarks;
    int hm_row_size = hm_width * hm_channels;
    int hm_pixel_size = hm_channels;

    for (int k = 0; k < kNumPoseLandmarks; ++k) {
      int center_col = object.landmarks3d[k].x * hm_width / input_w_;
      int center_row = object.landmarks3d[k].y * hm_height / input_h_; 
      if (center_col < 0 || center_col >= hm_width || 
          center_row < 0 || center_row >= hm_height) {
        continue;
      }
      int offset = (kernel_size_ - 1) / 2;
      int begin_col = std::max(0, center_col - offset);
      int end_col = std::min(hm_width, center_col + offset + 1);
      int begin_row = std::max(0, center_row - offset);
      int end_row = std::min(hm_height, center_row + offset + 1);

      float sum = 0;
      float weighted_col = 0;
      float weighted_row = 0;
      float max_confidence_value = 0;

      for (int row = begin_row; row < end_row; ++row) {
        for (int col = begin_col; col < end_col; ++col) {
          int idx = hm_row_size * row + hm_pixel_size * col + k;
          float confidence = sigmoid(headmap_ptr[idx]);
          sum += confidence;
          max_confidence_value = std::max(max_confidence_value, confidence);
          weighted_col += col * confidence;
          weighted_row += row * confidence;
        }
      }
      float x = weighted_col / hm_width / sum * input_w_;
      float y = weighted_row / hm_height / sum * input_h_;
      object.landmarks3d[k].x  = trans_[0] * x + trans_[1] * y + trans_[2];
      object.landmarks3d[k].y =  trans_[3] * x + trans_[4] * y + trans_[5];
    }
  }

  return true;
}

} // namespace mirror