/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:38
 * @LastEditTime: 2023-06-20 12:42:38
 * @Description: utils module
 * @FilePath: \Mediapipe-Hand\source\Utils.cc
 */
#include "Utils.h"

namespace mirror {
float ComputeRotation(const NormalizedLandmarkList &landmarks,
                      const std::pair<int, int> &image_size) {
  const float x0 = landmarks[kWristJoint].x * image_size.first;
  const float y0 = landmarks[kWristJoint].y * image_size.second;

  float x1 =
      (landmarks[kIndexFingerPIPJoint].x + landmarks[kRingFingerPIPJoint].x) /
      2.f;
  float y1 =
      (landmarks[kIndexFingerPIPJoint].y + landmarks[kRingFingerPIPJoint].y) /
      2.f;
  x1 = (x1 + landmarks[kMiddleFingerPIPJoint].x) / 2.f * image_size.first;
  y1 = (y1 + landmarks[kMiddleFingerPIPJoint].y) / 2.f * image_size.second;

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

} // namespace mirror
