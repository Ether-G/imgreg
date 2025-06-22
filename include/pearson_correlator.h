#pragma once

#include "correlation_result.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace imgreg {

class ThreadManager;
class MemoryPool;

class PearsonCorrelator {
public:
    PearsonCorrelator();
    ~PearsonCorrelator();
    
    // Main Pearson correlation method
    CorrelationResult correlate(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Configuration
    void setThreadCount(int thread_count);
    void setNormalizationType(const std::string& type); // "zscore", "minmax", "none"
    
    // Performance optimization
    void enableSIMD(bool enable);
    void setMemoryPool(std::shared_ptr<MemoryPool> pool);

private:
    // Core Pearson correlation methods
    double computePearsonCoefficient(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Statistical calculations
    double computeMean(const cv::Mat& image);
    double computeStandardDeviation(const cv::Mat& image, double mean);
    void computeMeanAndStd(const cv::Mat& image, double& mean, double& std_dev);
    
    // Normalization methods
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat zScoreNormalize(const cv::Mat& image);
    cv::Mat minMaxNormalize(const cv::Mat& image);
    
    // Threading support
    void processImageChunk(int start_row, int end_row, 
                          const cv::Mat& image,
                          double& partial_sum, double& partial_sum_sq);
    
    // SIMD optimized versions
    double computeMeanSIMD(const cv::Mat& image);
    double computeStdDevSIMD(const cv::Mat& image, double mean);
    double computeCorrelationSIMD(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Helper methods
    cv::Mat flattenImage(const cv::Mat& image);
    bool validateImageDimensions(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Configuration
    int thread_count_;
    std::string normalization_type_;
    bool simd_enabled_;
    
    // Performance tracking
    std::chrono::microseconds last_execution_time_;
    size_t last_memory_usage_;
    
    // Dependencies
    std::shared_ptr<ThreadManager> thread_manager_;
    std::shared_ptr<MemoryPool> memory_pool_;
    
    // Branch-free optimization helpers
    struct Statistics {
        double mean;
        double std_dev;
        double sum;
        double sum_sq;
        int count;
        
        Statistics() : mean(0.0), std_dev(0.0), sum(0.0), sum_sq(0.0), count(0) {}
    };
    
    Statistics computeStatisticsBranchless(const cv::Mat& image);
    
    // Memory optimization
    std::vector<double> temp_buffer_;
    void resizeTempBuffer(size_t size);
};

} // namespace imgreg 