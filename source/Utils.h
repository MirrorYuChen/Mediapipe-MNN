/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:31
 * @LastEditTime: 2023-08-01 18:23:58
 * @Description: utils module
 * @FilePath: \Mediapipe-MNN\source\Utils.h
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
constexpr int kNumPalmLandmarks = 21;
constexpr float kTargetPalmAngle = 90.0f;
constexpr int kNumFaceLandmarks = 478;
constexpr float kTargetFaceAngle = 0.0f;
float ComputeRotation(const Point2f & src, const Point2f &dst);

std::vector<Point2f> getInputRegion(const ImageHead &in, int out_w, int out_h, RotateType type);
std::vector<Point2f> getInputRegion(const ImageHead &in, RotateType type, const Rect &rect, int out_h, int out_w,
  float init_angle = 0.0f, float expand_scale = 1.0f, float offset_x_scale = 0.0f, float offset_y_scale = 0.0f);

float sigmoid(float x);

float getIouOfObjects(const ObjectInfo &a, const ObjectInfo &b);
void NMSObjects(std::vector<ObjectInfo> &objects, float iou_thresh);

float RotateTypeToAngle(RotateType type);

} // namespace mirror
