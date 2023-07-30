/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:38
 * @LastEditTime: 2023-07-30 12:53:37
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

std::vector<Point2f> getInputRegion(const ImageHead &in, RotateType type, const ObjectInfo &object,
                                    float expand_scale, float offset_x_scale, float offset_y_scale) {
  int width = in.width;
  int height = in.height;
  // 1.align the image
  float init_angle = object.angle;
  float angle = RotateTypeToAngle(type) + init_angle;

  // 2.get the align region
  CV::Matrix trans;
  trans.postRotate(angle, 0.5f * width, 0.5f * height);
  float rect_width = object.br.x - object.tl.x;
  float rect_height = object.br.y - object.tl.y;

  Point2f center;
  center.x = 0.5f * (object.br.x + object.tl.x);
  center.y = 0.5f * (object.br.y + object.tl.y);
  float center_x = trans[0] * center.x + trans[1] * center.y + trans[2] + offset_x_scale * rect_width;
  float center_y = trans[3] * center.x + trans[4] * center.y + trans[5] + offset_y_scale * rect_height;

  // 3. expand the region
  float half_max_side = MAX_(rect_width, rect_height) * 0.5f * expand_scale;
  float xmin = center_x - half_max_side;
  float ymin = center_y - half_max_side;
  float xmax = center_x + half_max_side;
  float ymax = center_y + half_max_side;

  std::vector<Point2f> region(4);
  region[0].x = xmin;
  region[0].y = ymin;
  region[1].x = xmin;
  region[1].y = ymax;
  region[2].x = xmax;
  region[2].y = ymin;
  region[3].x = xmax;
  region[3].y = ymax;

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

} // namespace mirror
