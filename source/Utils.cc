/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:38
 * @LastEditTime: 2023-06-25 17:44:04
 * @Description: utils module
 * @FilePath: \Mediapipe-Hand\source\Utils.cc
 */
#include "Utils.h"
#include <iostream>
#include <algorithm>

namespace mirror {
float ComputeRotation(const Point2f &src, const Point2f &dst) {
  const float dx = dst.x - src.x;
  const float dy = dst.y - src.y;
  const float angle = 90.0f - std::atan2(dy, dx) / M_PI * 180.0f;
  return -angle;
}

std::vector<Point2f> getInputRegion(int in_w, int in_h, int out_w, int out_h, RotateType type) {
  std::vector<Point2f> input_region(4);
  float in_scale = static_cast<float>(in_w) / static_cast<float>(in_h);
  float out_scale = static_cast<float>(out_w) / static_cast<float>(out_h);

  int region_w = in_w;
  int region_h = in_h;
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

float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

float getIouOfObjects(const ObjectInfo &a, const ObjectInfo &b) {
  float xmin = MAX_(a.tl.x, b.tl.x);
  float ymin = MAX_(a.tl.y, b.tl.y);
  float xmax = MIN_(a.br.x, b.br.x);
  float ymax = MIN_(a.br.y, b.br.y);

  float width = MAX_(0.0f, xmax - xmin);
  float height = MAX_(0.0f, ymax - ymin);
  float area_inter = width * height;
  float area_a = (a.br.x - a.tl.x) * (a.br.y - a.tl.y);
  float area_b = (b.br.x - b.tl.x) * (b.br.y - b.tl.y);

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

} // namespace mirror
