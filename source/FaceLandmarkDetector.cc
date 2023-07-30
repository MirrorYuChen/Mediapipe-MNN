/*
 * @Author: chenjingyu
 * @Date: 2023-07-30 12:58:06
 * @LastEditTime: 2023-07-30 17:03:35
 * @Description: face landmark detector
 * @FilePath: \Mediapipe-MNN\source\FaceLandmarkDetector.cc
 */
#include "FaceLandmarkDetector.h"
#include "Utils.h"
#include <cfloat>
#include <iostream>

namespace mirror {
using namespace MNN;
FaceLandmarkDetector::~FaceLandmarkDetector() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool FaceLandmarkDetector::LoadModel(const char *model_file) {
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

void FaceLandmarkDetector::setSourceFormat(int format) {
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

bool FaceLandmarkDetector::Detect(const ImageHead &in, RotateType type,
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
    std::vector<Point2f> region = getInputRegion(in, type, object, 1.5f, 0.0f, 0.0f);
    // clang-format off
    float points_src[] = {
      region[0].x, region[0].y,
      region[1].x, region[1].y,
      region[2].x, region[2].y,
      region[3].x, region[3].y,
    };
    float points_dst[] = {
      0.0f,  0.0f,
      0.0f, (float)(input_h_ - 1),
      (float)(input_w_ - 1), 0.0f,
      (float)(input_w_ - 1), (float)(input_h_ - 1),
    };
    // clang-format on
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
    MNN::Tensor *mesh = net_->getSessionOutput(sess_, "Identity");
    MNN::Tensor *face_score = net_->getSessionOutput(sess_, "Identity_1");
    //MNN::Tensor *tongue_score = net_->getSessionOutput(sess_, "Identity_2");

    std::shared_ptr<MNN::Tensor> output_mesh( 
        new MNN::Tensor(mesh, mesh->getDimensionType()));
    std::shared_ptr<MNN::Tensor> output_face_score(
        new MNN::Tensor(face_score, face_score->getDimensionType()));
    //std::shared_ptr<MNN::Tensor> output_tongue_score(
    //    new MNN::Tensor(tongue_score, tongue_score->getDimensionType()));

    mesh->copyToHostTensor(output_mesh.get());
    face_score->copyToHostTensor(output_face_score.get());
    //tongue_score->copyToHostTensor(output_tongue_score.get());

    Point2f landmark;
    object.landmarks3d.resize(478);
    float *mesh_ptr = output_mesh->host<float>();
    for (int k = 0; k < 478; ++k) {
      landmark.x = mesh_ptr[3 * k + 0];
      landmark.y = mesh_ptr[3 * k + 1];
      object.landmarks3d[k].x =
          trans_[0] * landmark.x + trans_[1] * landmark.y + trans_[2];
      object.landmarks3d[k].y =
          trans_[3] * landmark.x + trans_[4] * landmark.y + trans_[5];
      object.landmarks3d[k].z = mesh_ptr[3 * k + 2];
    }

    float *face_score_ptr = output_face_score->host<float>();
    object.score = sigmoid(face_score_ptr[0]);
  }

  return true;
}

} // namespace mirror
