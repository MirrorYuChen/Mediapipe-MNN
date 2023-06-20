/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:31
 * @LastEditTime: 2023-06-20 12:42:48
 * @Description: utils module
 * @FilePath: \Mediapipe-Hand\source\Utils.h
 */
#pragma once

#include "TypeDefines.h"
#include <cmath>
#include <utility>

namespace mirror {
#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

// Indices within the partial landmarks.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr int kNumLandmarks = 21;
constexpr float kTargetAngle = M_PI * 0.5f;

inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

float ComputeRotation(const NormalizedLandmarkList &landmarks,
                      const std::pair<int, int> &image_size);

int NormalizedLandmarkListToRect(const NormalizedLandmarkList &landmarks,
                                 const std::pair<int, int> &image_size,
                                 NormalizedRect &rect);

} // namespace mirror
