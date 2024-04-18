#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

///// Constant Parameters
const float kMinDepth = 1e-6;
const float kMaxDepth = 30.0;
const std::string cam_frame_id = "habitat_camera";
const float depth_scale = 25.5f; // 10.0 / (10.0 / 255.0)

ros::Publisher color_cloud_pub, semantic_cloud_pub;

Eigen::Matrix3f camera_intrinsics;

///////////////////////////////////////////////////////
///// *************** Publisher ************* ////////
//////////////////////////////////////////////////////
template <typename PointT>
void PublishCloud(const ros::Publisher &pcl_pub, 
                  const pcl::PointCloud<PointT> &cloud, 
                  const ros::Time timestamp, 
                  const std::string frame_id) {
  if (cloud.empty()) return;
  sensor_msgs::PointCloud2 pcl_msg;
  pcl::toROSMsg(cloud, pcl_msg);
  pcl_msg.header.stamp = timestamp;
  pcl_msg.header.frame_id = frame_id;
  pcl_pub.publish(pcl_msg);
}

void CreateDepthCloud(PointCloudT::Ptr& depth_cloud, 
                      const cv::Mat& depth_image_float, 
                      const cv::Mat& color_image) {
  if (!color_image.empty()) {
    depth_cloud->reserve(depth_image_float.rows * depth_image_float.cols);
    for (int y = 0; y < depth_image_float.rows; y++) {
      for (int x = 0; x < depth_image_float.cols; x++) {
        const float& depth = depth_image_float.at<float>(y, x);
        if (depth < kMinDepth) continue;
        Eigen::Vector3f pts_cam;
        pts_cam[2] = depth;
        pts_cam[0] = depth * (static_cast<float>(x) - camera_intrinsics(0, 2)) / camera_intrinsics(0, 0);
        pts_cam[1] = depth * (static_cast<float>(y) - camera_intrinsics(1, 2)) / camera_intrinsics(1, 1);
        PointT point;
        point.x = pts_cam.x();
        point.y = pts_cam.y();
        point.z = pts_cam.z();
        const cv::Vec3b& color = color_image.at<cv::Vec3b>(y, x);
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];
        depth_cloud->push_back(point);
      }
    }
  }
}

void ProcessDepthImage(const cv::Mat &src, cv::Mat &dst_float) {
  // NOTE(gogojjh): address habitat depth image
  dst_float = src;

  // dst_float = cv::Mat(src.rows, src.cols, CV_32FC1, cv::Scalar(0.0));
  // for (int y = 0; y < src.rows; y++) {
  //   for (int x = 0; x < src.cols; x++) {
  //     dst_float.at<float>(y, x) = static_cast<float>(src.at<uint16_t>(y, x)) / depth_scale;
  //   }
  // }
}

void ProcessRGBDImage(PointCloudT::Ptr &depth_cloud,
                      const cv::Mat &depth_image_float,
                      const cv::Mat &color_image) {
  if (!color_image.empty()) {
    cv::Mat img_und = color_image;
    // if (distort_image) {
    //   img_und = camera_ptr->undistortImage(color_image);
    // } else {
    //   img_und = color_image;
    // }
    CreateDepthCloud(depth_cloud, depth_image_float, img_und);
  }
}

// *************************************************************************** //
// ***************************** Callback Functions ************************** //
// ******************************************^^^^***************************** //
void ColorDepthImageCallback(const sensor_msgs::ImageConstPtr& color_msg,
                             const sensor_msgs::ImageConstPtr& depth_msg) {
  try {
    cv::Mat depth_image =
        cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)
            ->image;
    cv::Mat color_image =
        cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8)
            ->image;
    ros::Time timestamp = depth_msg->header.stamp;

    cv::Mat depth_image_float;
    ProcessDepthImage(depth_image, depth_image_float);
    PointCloudT::Ptr depth_cloud(new PointCloudT);
    ProcessRGBDImage(depth_cloud, depth_image_float, color_image);

    PublishCloud(color_cloud_pub, *depth_cloud, timestamp, cam_frame_id);
    // std::cout << "[ColorDepthImageCallback] Publish " << depth_cloud->size() << " RGB points" << std::endl;
  } catch (cv_bridge::Exception& e) {
    std::cout << "cv_bridge exception: " << e.what() << std::endl;
  }
}

void SemDepthImageCallback(const sensor_msgs::ImageConstPtr& sem_msg,
                           const sensor_msgs::ImageConstPtr& depth_msg) {
  try {
    cv::Mat depth_image =
        cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1)
            ->image;
    cv::Mat sem_image =
        cv_bridge::toCvCopy(sem_msg, sensor_msgs::image_encodings::BGR8)
            ->image;
    ros::Time timestamp = depth_msg->header.stamp;

    cv::Mat depth_image_float;
    ProcessDepthImage(depth_image, depth_image_float);
    PointCloudT::Ptr depth_cloud(new PointCloudT);
    ProcessRGBDImage(depth_cloud, depth_image_float, sem_image);

    PublishCloud(semantic_cloud_pub, *depth_cloud, timestamp, cam_frame_id);
    // std::cout << "[SemDepthImageCallback] Publish " << depth_cloud->size() << " Sem points" << std::endl;
  } catch (cv_bridge::Exception& e) {
    std::cout << "cv_bridge exception: " << e.what() << std::endl;
  }
}

// *********************************************************************** //
// ***************************** Main Functions ************************** //
// *********************************************************************** //
int main(int argc, char** argv) {
  ros::init(argc, argv, "publish_pts_from_depth");
  ros::NodeHandle nh, nh_private("~");

  // clang-format off
  const float width = 640.0f;
  const float height = 360.0f;
  const float hfov = 114.591560981; // deg
  const float f = width / (2.0 * tan(hfov / 2.0 * M_PI / 180.0));
  const float cx = width / 2.0;
  const float cy = height / 2.0;
  camera_intrinsics << f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0;
  std::cout << "camera_intrinsics: \n" << camera_intrinsics << std::endl;

  // Create a publisher for the point cloud
  color_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/habitat_camera/color_pointcloud", 100);
  semantic_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/habitat_camera/semantic_pointcloud", 100);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RGBDSyncPolicy;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/habitat_camera/depth/image", 10);
  message_filters::Subscriber<sensor_msgs::Image> color_sub(nh, "/habitat_camera/color/image", 10);
  message_filters::Subscriber<sensor_msgs::Image> sem_sub(nh, "/habitat_camera/semantic/image", 10);
  message_filters::Synchronizer<RGBDSyncPolicy> color_depth_sync(RGBDSyncPolicy(100), color_sub, depth_sub);
  message_filters::Synchronizer<RGBDSyncPolicy> sem_depth_sync(RGBDSyncPolicy(100), sem_sub, depth_sub);
  color_depth_sync.registerCallback(boost::bind(&ColorDepthImageCallback, _1, _2));
  sem_depth_sync.registerCallback(boost::bind(&SemDepthImageCallback, _1, _2));
  ros::spin();
  // clang-format on

  return 0;
}
