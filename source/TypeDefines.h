/*
 * @Author: chenjingyu
 * @Date: 2023-06-19 17:18:41
 * @LastEditTime: 2023-08-20 01:17:26
 * @Description: type defines
 * @FilePath: \Mediapipe-MNN\source\TypeDefines.h
 */
#pragma once

#include <string>
#include <vector>
namespace mirror {
using byte = unsigned char;
struct Point2f {
  float x;
  float y;
};

struct Point3f {
  float x;
  float y;
  float z;
  float visibility;  // landmark score
  float presence;     // landmark in frame score
};

using LandmarkList = std::vector<Point2f>;
using NormalizedLandmarkList = std::vector<Point2f>;

struct Rect {
  float left;
  float top;
  float right;
  float bottom;
};

struct ObjectInfo {
  Rect rect;
  std::vector<Point2f> landmarks;
  std::vector<Point3f> landmarks3d;
  float score;
  float angle;
  int left_right;
};

enum PixelFormat {
  RGBA = 0,
  RGB = 1,
  BGR = 2,
  GRAY = 3,
  BGRA = 4,
  YCrCb = 5,
  YUV = 6,
  HSV = 7,
  XYZ = 8,
  BGR555 = 9,
  BGR565 = 10,
  YUV_NV21 = 11,
  YUV_NV12 = 12,
  YUV_I420 = 13,
  HSV_FULL = 14,
};

enum RotateType {
  CLOCKWISE_ROTATE_0 = 0, ///< 图像不需要旋转,图像中的人脸为正脸
  CLOCKWISE_ROTATE_90 = 1, ///< 图像需要顺时针旋转90度,使图像中的人脸为正
  CLOCKWISE_ROTATE_180 = 2, ///< 图像需要顺时针旋转180度,使图像中的人脸为正
  CLOCKWISE_ROTATE_270 = 3 ///< 图像需要顺时针旋转270度,使图像中的人脸为正
};

typedef struct ImageHead_t {
  byte *data;               ///< 图像数据指针
  PixelFormat pixel_format; ///< 像素格式
  int width;                ///< 宽度(以像素为单位)
  int height;               ///< 高度(以像素为单位)
  int width_step;           ///< 跨度, 即每行所占的字节数
  double time_stamp;        ///< 时间戳
} ImageHead;

struct Embedding {
  std::vector<float> float_embedding;
  std::string quantized_embedding;
};

struct ClassifierInfo {
  int id;
  float score;
};

} // namespace mirror
