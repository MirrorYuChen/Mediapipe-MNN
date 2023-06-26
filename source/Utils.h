/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:31
 * @LastEditTime: 2023-06-25 17:44:25
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
constexpr int kNumLandmarks = 21;
constexpr float kTargetAngle = M_PI * 0.5f;
float ComputeRotation(const Point2f & src, const Point2f &dst);

std::vector<Point2f> getInputRegion(int in_w, int in_h, int out_w, int out_h, RotateType type);

float sigmoid(float x);

float getIouOfObjects(const ObjectInfo &a, const ObjectInfo &b);
void NMSObjects(std::vector<ObjectInfo> &objects, float iou_thresh);

float RotateTypeToAngle(RotateType type);
} // namespace mirror
