/*
 * @Author: chenjingyu
 * @Date: 2023-08-02 12:43:33
 * @LastEditTime: 2023-08-19 22:33:14
 * @Description: Face detector YN
 * @FilePath: \Mediapipe-MNN\source\FaceDetectorYN.cc
 */
#include "FaceDetectorYN.h"

#include "Utils.h"
#include <iostream>

using namespace MNN;
namespace mirror {
FaceDetectorYN::~FaceDetectorYN() {
  net_->releaseModel();
  net_->releaseSession(sess_);
}

bool FaceDetectorYN::LoadModel(const char *model_file) {
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

void FaceDetectorYN::setFormat(int format) {
  // create image process
  CV::ImageProcess::Config image_process_config;
  image_process_config.filterType = CV::BILINEAR;
  image_process_config.sourceFormat = CV::ImageFormat(format);
  image_process_config.destFormat = CV::BGR;
  image_process_config.wrap = CV::ZERO;
  pretreat_ = std::shared_ptr<CV::ImageProcess>(
      CV::ImageProcess::create(image_process_config));
}

bool FaceDetectorYN::Detect(const ImageHead &in, RotateType type,
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
  const size_t stride_num = strides_.size();
  ObjectInfo object;
  for (size_t i = 0; i < stride_num; ++i) {
    int cols = input_w_ / strides_[i];
    int rows = input_h_ / strides_[i];

    MNN::Tensor *cls  = net_->getSessionOutput(sess_, output_names_[i].c_str());
    MNN::Tensor *obj  = net_->getSessionOutput(sess_, output_names_[i + stride_num].c_str());
    MNN::Tensor *bbox = net_->getSessionOutput(sess_, output_names_[i + stride_num * 2].c_str());
    MNN::Tensor *kps = net_->getSessionOutput(sess_, output_names_[i + stride_num * 3].c_str());

    std::shared_ptr<MNN::Tensor> cls_result(new MNN::Tensor(cls, cls->getDimensionType()));
    std::shared_ptr<MNN::Tensor> obj_result(new MNN::Tensor(obj, obj->getDimensionType()));
    std::shared_ptr<MNN::Tensor> bbox_result(new MNN::Tensor(bbox, bbox->getDimensionType()));
    std::shared_ptr<MNN::Tensor> kps_result(new MNN::Tensor(kps, kps->getDimensionType()));

    cls->copyToHostTensor(cls_result.get());
    obj->copyToHostTensor(obj_result.get());
    bbox->copyToHostTensor(bbox_result.get());
    kps->copyToHostTensor(kps_result.get());

    float *cls_ptr = cls_result->host<float>();
    float *obj_ptr = obj_result->host<float>();
    float *bbox_ptr = bbox_result->host<float>();
    float *kps_ptr = kps_result->host<float>();

    for(int r = 0; r < rows; ++r) {
      for(int c = 0; c < cols; ++c) {
        size_t idx = r * cols + c;

        // score
        float cls_score = cls_ptr[idx];
        float obj_score = obj_ptr[idx];

        // Clamp
        cls_score = MIN_(cls_score, 1.0f);
        cls_score = MAX_(cls_score, 0.0f);
        obj_score = MIN_(obj_score, 1.0f);
        obj_score = MAX_(obj_score, 0.0f);
        float score = std::sqrt(cls_score * obj_score);
        if (score < score_thresh_) continue;
        object.score = score;

        // bbox
        float cx = ((c + bbox_ptr[idx * 4 + 0]) * strides_[i]);
        float cy = ((r + bbox_ptr[idx * 4 + 1]) * strides_[i]);
        float w = exp(bbox_ptr[idx * 4 + 2]) * strides_[i];
        float h = exp(bbox_ptr[idx * 4 + 3]) * strides_[i];

        Point2f tl, br, tl_origin, br_origin;
        tl.x = (cx - 0.5f * w);
        tl.y = (cy - 0.5f * h);
        br.x = (cx + 0.5f * w);
        br.y = (cy + 0.5f * h);

        object.landmarks.resize(5);
        for (int k = 0; k < 5; ++k) {
          float x = (kps_ptr[10 * idx + 2 * k] + c) * strides_[i];
          float y = (kps_ptr[10 * idx + 2 * k + 1] + r) * strides_[i];
          object.landmarks[k].x = trans_[0] * x + trans_[1] * y + trans_[2];
          object.landmarks[k].y = trans_[3] * x + trans_[4] * y + trans_[5];
        }

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