/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:38
 * @LastEditTime: 2023-08-04 22:43:01
 * @Description: utils module
 * @FilePath: \Mediapipe-MNN\source\Utils.cc
 */
#include "Utils.h"
#include <iostream>
#include <algorithm>

#include <MNN/Matrix.h>
#include <MNN/Rect.h>

namespace mirror {
using namespace MNN;
float ComputeRotation(const Point2f &src, const Point2f &dst) {
  const float dx = dst.x - src.x;
  const float dy = dst.y - src.y;
  const float angle = 90.0f - std::atan2(dy, dx) / M_PI * 180.0f;
  return -angle;
}

std::vector<Point2f> getInputRegion(const ImageHead &in, int out_w, int out_h, RotateType type) {
  std::vector<Point2f> input_region(4);
  float in_scale = static_cast<float>(in.width) / static_cast<float>(in.height);
  float out_scale = static_cast<float>(out_w) / static_cast<float>(out_h);

  int region_w = in.width;
  int region_h = in.height;
  if (in_scale > out_scale) {
    region_h = region_w / out_scale;
  } else {
    region_w = region_h * out_scale;
  }
  
  switch (type) {
    case RotateType::CLOCKWISE_ROTATE_0:
      input_region[0].x = 0.0f,         input_region[0].y = 0.0f;
      input_region[1].x = 0.0f,         input_region[1].y = region_h - 1;
      input_region[2].x = region_w - 1, input_region[2].y = 0.0f;
      input_region[3].x = region_w - 1, input_region[3].y = region_h - 1;
      break;
    case RotateType::CLOCKWISE_ROTATE_90:
      input_region[2].x = 0.0f,         input_region[2].y = 0.0f;
      input_region[0].x = 0.0f,         input_region[0].y = region_h - 1;
      input_region[3].x = region_w - 1, input_region[3].y = 0.0f;
      input_region[1].x = region_w - 1, input_region[1].y = region_h - 1;
      break;
    case RotateType::CLOCKWISE_ROTATE_180:
      input_region[3].x = 0.0f,         input_region[3].y = 0.0f;
      input_region[2].x = 0.0f,         input_region[2].y = region_h - 1;
      input_region[1].x = region_w - 1, input_region[1].y = 0.0f;
      input_region[0].x = region_w - 1, input_region[0].y = region_h - 1;
      break;
    case RotateType::CLOCKWISE_ROTATE_270:
      input_region[1].x = 0.0f,         input_region[1].y = 0.0f;
      input_region[3].x = 0.0f,         input_region[3].y = region_h - 1;
      input_region[0].x = region_w - 1, input_region[0].y = 0.0f;
      input_region[2].x = region_w - 1, input_region[2].y = region_h - 1;
      break;
    default:
      std::cout << "Error unknown rotate type." << std::endl;
      break;
  }

  return input_region;
}

std::vector<Point2f> getInputRegion(const ImageHead &in, RotateType type,
                                    const Rect &rect, int out_h, int out_w,
                                    float init_angle, float expand_scale,
                                    float offset_x_scale, float offset_y_scale) {
  int width = in.width;
  int height = in.height;
  float out_scale = static_cast<float>(out_w) / static_cast<float>(out_h);
  // 1.align the image
  float angle = RotateTypeToAngle(type) + init_angle;

  // 2.get the align region
  CV::Matrix trans;
  trans.postRotate(angle, 0.5f * width, 0.5f * height);
  float rect_width = rect.right - rect.left;
  float rect_height = rect.bottom - rect.top;
  float center_x = 0.5f * (rect.left + rect.right);
  float center_y = 0.5f * (rect.top + rect.bottom);

  float crop_center_x = trans[0] * center_x + trans[1] * center_y + trans[2] + offset_x_scale * rect_width;
  float crop_center_y = trans[3] * center_x + trans[4] * center_y + trans[5] + offset_y_scale * rect_height;

  // 3. expand the region
  float half_max_side = MAX_(rect_width, rect_height) * 0.5f * expand_scale;
  float half_crop_width = half_max_side, half_crop_height = half_max_side;
  if (out_scale > 1.0f) {
    half_crop_width = half_max_side;
    half_crop_height = half_crop_width / out_scale;
  } else {
    half_crop_height = half_max_side;
    half_crop_width = half_crop_height * out_scale;
  }

  float xmin = crop_center_x - half_crop_width;
  float ymin = crop_center_y - half_crop_height;
  float xmax = crop_center_x + half_crop_width;
  float ymax = crop_center_y + half_crop_height;

  std::vector<Point2f> region(4);
  region[0].x = xmin, region[0].y = ymin;
  region[1].x = xmin, region[1].y = ymax;
  region[2].x = xmax, region[2].y = ymin;
  region[3].x = xmax, region[3].y = ymax;

  std::vector<Point2f> result(4);
  trans.invert(&trans);
  for (size_t i = 0; i < region.size(); ++i) {
    result[i].x = trans[0] * region[i].x + trans[1] * region[i].y + trans[2];
    result[i].y = trans[3] * region[i].x + trans[4] * region[i].y + trans[5];
  }
  return result;
}

float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

float getIouOfObjects(const ObjectInfo &a, const ObjectInfo &b) {
  float xmin = MAX_(a.rect.left, b.rect.left);
  float ymin = MAX_(a.rect.top, b.rect.top);
  float xmax = MIN_(a.rect.right, b.rect.right);
  float ymax = MIN_(a.rect.bottom, b.rect.bottom);

  float width = MAX_(0.0f, xmax - xmin);
  float height = MAX_(0.0f, ymax - ymin);
  float area_inter = width * height;
  float area_a = (a.rect.right - a.rect.left) * (a.rect.bottom - a.rect.top);
  float area_b = (b.rect.right - b.rect.left) * (b.rect.bottom - b.rect.top);

  float iou = area_inter / (area_a + area_b - area_inter);
  return (iou >= 0.0f ? iou : 0.0f);
}

void NMSObjects(std::vector<ObjectInfo> &objects, float iou_thresh) {
  std::sort(objects.begin(), objects.end(), [](const ObjectInfo &a, const ObjectInfo &b) {
    return a.score > b.score;
  });
  std::vector<bool> delete_flag(objects.size(), false);
  for (size_t i = 0; i < objects.size(); ++i) {
    if (delete_flag[i]) continue;
    for (size_t j = i + 1; j < objects.size(); ++j) {
      float iou = getIouOfObjects(objects[i], objects[j]);
      if (iou > iou_thresh) delete_flag[j] = true;
    }
  }

  std::vector<ObjectInfo> result;
  for (size_t i = 0; i < objects.size(); ++i) {
    if (delete_flag[i]) continue;
    result.emplace_back(objects[i]);
  }
  objects = result;
}

float RotateTypeToAngle(RotateType type) {
  float angle = 0.0f;
  switch (type) {
    case RotateType::CLOCKWISE_ROTATE_0:
      break;
    case RotateType::CLOCKWISE_ROTATE_90:
      angle = 90.0f;
      break;
    case RotateType::CLOCKWISE_ROTATE_180:
      angle = 180.0f;
      break;
    case RotateType::CLOCKWISE_ROTATE_270:
      angle = 270.0f;
      break;
  }
  return angle;
}

Embedding BuildFloatEmbedding(const std::vector<float> &values) {
  Embedding embedding;
  embedding.float_embedding = values;
  return embedding;
}

Embedding BuildQuantizedEmbedding(const std::vector<int8_t> &values) {
  Embedding embedding;
  const uint8_t* data = reinterpret_cast<const uint8_t*>(values.data());
  embedding.quantized_embedding = {data, data + values.size()};
  return embedding;  
}

optional<double> CosineSimilarity(const Embedding& u, const Embedding& v) {
  if (!u.float_embedding.empty() && !v.float_embedding.empty()) {
    if (u.float_embedding.size() != v.float_embedding.size()) {
      return nullopt;
    }
    return ComputeCosineSimilarity(
      u.float_embedding.data(),
      v.float_embedding.data(),
      u.float_embedding.size()
    );
  }

  if (!u.quantized_embedding.empty() && !v.quantized_embedding.empty()) {
    if (u.quantized_embedding.size() != v.quantized_embedding.size()) {
      return nullopt;
    }
    return ComputeCosineSimilarity(
      reinterpret_cast<const int8_t*>(u.quantized_embedding.data()),
      reinterpret_cast<const int8_t*>(v.quantized_embedding.data()),
      u.quantized_embedding.size()
    );
  }
  return nullopt;
}

float GetInverseL2Norm(const float *values, int size) {
  float squared_l2_norm = 0.0f;
  for (int i = 0; i < size; ++i) {
    squared_l2_norm += values[i] * values[i];
  }
  float inv_l2_norm = 1.0f;
  if (squared_l2_norm > 0.0f) {
    inv_l2_norm = 1.0f / std::sqrt(squared_l2_norm);
  }
  return inv_l2_norm;
}

Embedding FillFloatEmbedding(const float *data, int size, bool l2_normalize) {
  float inv_l2_norm =
      l2_normalize ? GetInverseL2Norm(data, size) : 1.0f;
  Embedding result;
  for (int i = 0; i < size; ++i) {
    result.float_embedding.emplace_back(data[i] * inv_l2_norm);
  }
  return result;
}

Embedding FillQuantizedEmbedding(const float *data, int size, bool l2_normalize) {
  float inv_l2_norm =
      l2_normalize ? GetInverseL2Norm(data, size) : 1.0f;
  Embedding result;
  result.quantized_embedding.resize(size);
  for (int i = 0; i < size; ++i) {
    // Normalize.
    float normalized = data[i] * inv_l2_norm;
    // Quantize.
    int unclamped_value = static_cast<int>(roundf(normalized * 128));
    // Clamp and assign.
    result.quantized_embedding[i] =
        static_cast<char>(std::max(-128, std::min(unclamped_value, 127)));
  }
  return result;
}

} // namespace mirror
