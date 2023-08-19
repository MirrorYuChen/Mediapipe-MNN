/*
 * @Author: chenjingyu
 * @Date: 2023-06-20 12:29:31
 * @LastEditTime: 2023-08-20 01:12:15
 * @Description: utils module
 * @FilePath: \Mediapipe-MNN\source\Utils.h
 */
#pragma once

#include "TypeDefines.h"
#include "optional.h"
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
constexpr int kNumPoseLandmarks = 39;
float ComputeRotation(const Point2f &src, const Point2f &dst);

std::vector<Point2f> getInputRegion(const ImageHead &in, RotateType type,
                                    int out_w, int out_h,
                                    bool keep_aspect = true);
std::vector<Point2f> getInputRegion(const ImageHead &in, RotateType type,
                                    const Rect &rect, int out_h, int out_w,
                                    float init_angle = 0.0f,
                                    float expand_scale = 1.0f,
                                    float offset_x_scale = 0.0f,
                                    float offset_y_scale = 0.0f);

std::vector<Point2f>
getInputRegion(const ImageHead &in, RotateType type,
               const ObjectInfo &object, float expand_scale = 1.0f);

float sigmoid(float x);

float getIouOfObjects(const ObjectInfo &a, const ObjectInfo &b);
void NMSObjects(std::vector<ObjectInfo> &objects, float iou_thresh);

float RotateTypeToAngle(RotateType type);

template <typename T>
optional<double> ComputeCosineSimilarity(const T &u, const T &v,
                                         int num_elements) {
  if (num_elements <= 0) {
    return nullopt;
  }
  double dot_product = 0.0;
  double norm_u = 0.0;
  double norm_v = 0.0;
  for (int i = 0; i < num_elements; ++i) {
    dot_product += u[i] * v[i];
    norm_u += u[i] * u[i];
    norm_v += v[i] * v[i];
  }
  if (norm_u <= 0.0 || norm_v <= 0.0) {
    return nullopt;
  }
  return dot_product / std::sqrt(norm_u * norm_v);
}

Embedding BuildFloatEmbedding(const std::vector<float> &values);
Embedding BuildQuantizedEmbedding(const std::vector<int8_t> &values);
optional<double> CosineSimilarity(const Embedding &u, const Embedding &v);

float GetInverseL2Norm(const float *values, int size);

Embedding FillFloatEmbedding(const float *data, int size, bool l2_normalize);

Embedding FillQuantizedEmbedding(const float *data, int size,
                                 bool l2_normalize);

} // namespace mirror
