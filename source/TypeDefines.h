/*
 * @Author: chenjingyu
 * @Date: 2023-06-19 17:18:41
 * @LastEditTime: 2023-06-20 12:41:14
 * @Description: type defines
 * @FilePath: \Mediapipe-Hand\source\TypeDefines.h
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

using NormalizedLandmarkList = std::vector<Point2f>;

struct Rect {
    float cx;
    float cy;
    float w;
    float h;
    float r;
};

using NormalizedRect = Rect;

struct ObjectInfo {
    std::string name;
    Point2f tl;
    Point2f br;
    Point2f landmarks[7];
    float score;
};

enum PixelFormat {
    AISDK_PIX_FMT_GRAY8 = 1,   ///< Y    1        8bpp ( 单通道8bit灰度像素 )
    AISDK_PIX_FMT_YUV420P, ///< YUV  4:2:0   12bpp ( 3通道, 一个亮度通道, 另两个为U分量和V分量通道, 所有通道都是连续的
    AISDK_PIX_FMT_NV12,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为UV分量交错 )
    AISDK_PIX_FMT_NV21,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为VU分量交错 )
    AISDK_PIX_FMT_BGRA8888,///< BGRA 8:8:8:8 32bpp ( 4通道32bit BGRA 像素 )
    AISDK_PIX_FMT_BGR888,  ///< BGR  8:8:8   24bpp ( 3通道24bit BGR 像素 )
    AISDK_PIX_FMT_RGBA8888,///< RGBA 8:8:8:8 32bpp ( 4通道32bit RGBA 像素 )
    AISDK_PIX_FMT_RGB888   ///< RGB  8:8:8   24bpp ( 3通道24bit RGB 像素 )
};

enum RotateType {
    AISDK_CLOCKWISE_ROTATE_0 = 0,  ///< 图像不需要旋转,图像中的人脸为正脸
    AISDK_CLOCKWISE_ROTATE_90 = 1, ///< 图像需要顺时针旋转90度,使图像中的人脸为正
    AISDK_CLOCKWISE_ROTATE_180 = 2,///< 图像需要顺时针旋转180度,使图像中的人脸为正
    AISDK_CLOCKWISE_ROTATE_270 = 3 ///< 图像需要顺时针旋转270度,使图像中的人脸为正    
};

typedef struct ImageHead_t {
    byte* data;               ///< 图像数据指针
    PixelFormat pixel_format; ///< 像素格式
    int width;                ///< 宽度(以像素为单位)
    int height;               ///< 高度(以像素为单位)
    int width_step;           ///< 跨度, 即每行所占的字节数
    double time_stamp;        ///< 时间戳
} ImageHead;


} // namespace mirror

