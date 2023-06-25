/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:38
 * @LastEditTime: 2023-06-25 11:07:58
 * @Description: utils module
 * @FilePath: \Mediapipe-Hand\source\Utils.cc
 */
#include "Utils.h"
#include <iostream>
#include <algorithm>

namespace mirror {
float ComputeRotation(const NormalizedLandmarkList &landmarks,
                      const std::pair<int, int> &image_size) {
  const float x0 = landmarks[kWristJoint].x * image_size.first;
  const float y0 = landmarks[kWristJoint].y * image_size.second;

  float x1 =
      (landmarks[kIndexFingerJoint].x + landmarks[kRingFingerJoint].x) /
      2.f;
  float y1 =
      (landmarks[kIndexFingerJoint].y + landmarks[kRingFingerJoint].y) /
      2.f;
  x1 = (x1 + landmarks[kMiddleFingerJoint].x) / 2.f * image_size.first;
  y1 = (y1 + landmarks[kMiddleFingerJoint].y) / 2.f * image_size.second;

  const float rotation =
      NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
  return rotation;
}

int NormalizedLandmarkListToRect(const NormalizedLandmarkList &landmarks,
                                 const std::pair<int, int> &image_size,
                                 NormalizedRect &rect) {
  const float rotation = ComputeRotation(landmarks, image_size);
  const float reverse_angle = NormalizeRadians(-rotation);

  // Find boundaries of landmarks.
  float max_x = std::numeric_limits<float>::min();
  float max_y = std::numeric_limits<float>::min();
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  for (int i = 0; i < landmarks.size(); ++i) {
    max_x = std::max(max_x, landmarks[i].x);
    max_y = std::max(max_y, landmarks[i].y);
    min_x = std::min(min_x, landmarks[i].x);
    min_y = std::min(min_y, landmarks[i].y);
  }
  const float axis_aligned_center_x = (max_x + min_x) / 2.f;
  const float axis_aligned_center_y = (max_y + min_y) / 2.f;

  // Find boundaries of rotated landmarks.
  max_x = std::numeric_limits<float>::min();
  max_y = std::numeric_limits<float>::min();
  min_x = std::numeric_limits<float>::max();
  min_y = std::numeric_limits<float>::max();
  for (int i = 0; i < landmarks.size(); ++i) {
    const float original_x =
        (landmarks[i].x - axis_aligned_center_x) * image_size.first;
    const float original_y =
        (landmarks[i].y - axis_aligned_center_y) * image_size.second;

    const float projected_x = original_x * std::cos(reverse_angle) -
                              original_y * std::sin(reverse_angle);
    const float projected_y = original_x * std::sin(reverse_angle) +
                              original_y * std::cos(reverse_angle);

    max_x = std::max(max_x, projected_x);
    max_y = std::max(max_y, projected_y);
    min_x = std::min(min_x, projected_x);
    min_y = std::min(min_y, projected_y);
  }
  const float projected_center_x = (max_x + min_x) / 2.f;
  const float projected_center_y = (max_y + min_y) / 2.f;

  const float center_x = projected_center_x * std::cos(rotation) -
                         projected_center_y * std::sin(rotation) +
                         image_size.first * axis_aligned_center_x;
  const float center_y = projected_center_x * std::sin(rotation) +
                         projected_center_y * std::cos(rotation) +
                         image_size.second * axis_aligned_center_y;
  const float width = (max_x - min_x) / image_size.first;
  const float height = (max_y - min_y) / image_size.second;

  rect.cx = center_x / image_size.first;
  rect.cy = center_y / image_size.second;
  rect.w = width;
  rect.h = height;
  rect.h = rotation;

  return 0;
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
