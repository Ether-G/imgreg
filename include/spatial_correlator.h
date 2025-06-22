#pragma once

#include "correlation_result.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace imgreg {

class ThreadManager;
class MemoryPool;

class SpatialCorrelator {
public:
    SpatialCorrelator();
    ~SpatialCorrelator();
    
    // Main correlation method using sliding window
    CorrelationResult correlate(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Configuration
    void setThreadCount(int thread_count);
    void setWindowSize(int width, int height);
    void setStepSize(int step_x, int step_y);
    
    // Performance optimization
    void enableSIMD(bool enable);
    void setMemoryPool(std::shared_ptr<MemoryPool> pool);

private:
    // Core correlation methods
    double computeNormalizedCrossCorrelation(const cv::Mat& template_img, 
                                            const cv::Mat& search_img,
                                            int offset_x, int offset_y);
    
    std::pair<int, int> findPeakLocation(const std::vector<std::vector<double>>& correlation_map);
    
    // Threading support
    void processWindowRange(int start_x, int end_x, int start_y, int end_y,
                           const cv::Mat& image_a, const cv::Mat& image_b,
                           std::vector<std::vector<double>>& correlation_map);
    
    // SIMD optimized versions
    double computeCorrelationSIMD(const cv::Mat& template_img, 
                                 const cv::Mat& search_img,
                                 int offset_x, int offset_y);
    
    // Helper methods
    cv::Mat extractWindow(const cv::Mat& image, int x, int y, int width, int height);
    void normalizeWindow(cv::Mat& window);
    
    // Configuration
    int thread_count_;
    int window_width_;
    int window_height_;
    int step_x_;
    int step_y_;
    bool simd_enabled_;
    
    // Performance tracking
    std::chrono::microseconds last_execution_time_;
    size_t last_memory_usage_;
    
    // Dependencies
    std::shared_ptr<ThreadManager> thread_manager_;
    std::shared_ptr<MemoryPool> memory_pool_;
    
    // Branch-free optimization helpers
    struct CorrelationWindow {
        int x, y;
        double correlation_value;
        
        CorrelationWindow() : x(0), y(0), correlation_value(0.0) {}
        CorrelationWindow(int x_, int y_, double val) : x(x_), y(y_), correlation_value(val) {}
    };
    
    CorrelationWindow findPeakBranchless(const std::vector<std::vector<double>>& correlation_map);
};

} // namespace imgreg 