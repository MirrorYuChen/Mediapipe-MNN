/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:31
 * @LastEditTime: 2023-06-21 09:58:27
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

#define MAX_(x, y) ((x) > (y) ? (x) : y)
#define MIN_(x, y) ((x) < (y) ? (x) : y)

// Indices within the partial landmarks.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerJoint = 6;
constexpr int kIndexFingerJoint = 4;
constexpr int kRingFingerJoint = 8;
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

std::vector<Point2f> getInputRegion(int in_w, int in_h, int out_w, int out_h, RotateType type);

} // namespace mirror
