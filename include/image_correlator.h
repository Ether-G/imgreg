#pragma once

#include "correlation_result.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace imgreg {

class SpatialCorrelator;
class FFTCorrelator;
class PearsonCorrelator;
class PerformanceAnalyzer;
class ThreadManager;
class MemoryPool;

class ImageCorrelator {
public:
    ImageCorrelator();
    ~ImageCorrelator();
    
    // Main correlation interface
    CorrelationResult correlate(const cv::Mat& image_a, const cv::Mat& image_b, 
                               const std::string& method = "all");
    
    // Individual method calls
    CorrelationResult spatialCorrelate(const cv::Mat& image_a, const cv::Mat& image_b);
    CorrelationResult fftCorrelate(const cv::Mat& image_a, const cv::Mat& image_b);
    CorrelationResult pearsonCorrelate(const cv::Mat& image_a, const cv::Mat& image_b);
    
    // Performance benchmarking
    std::vector<CorrelationResult> benchmarkAllMethods(const cv::Mat& image_a, 
                                                       const cv::Mat& image_b,
                                                       int num_threads = 1);
    
    // Configuration
    void setThreadCount(int thread_count);
    void setImageSize(int width, int height);
    void enablePerformanceProfiling(bool enable);
    
    // Results and analysis
    PerformanceMetrics getPerformanceMetrics() const;
    void exportResults(const std::string& filename) const;
    void generateReport(const std::string& filename) const;

private:
    // Core components
    int thread_count_;
    int image_width_;
    int image_height_;
    bool performance_profiling_enabled_;
    std::unique_ptr<SpatialCorrelator> spatial_correlator_;
    std::unique_ptr<FFTCorrelator> fft_correlator_;
    std::unique_ptr<PearsonCorrelator> pearson_correlator_;
    std::unique_ptr<PerformanceAnalyzer> performance_analyzer_;
    std::unique_ptr<ThreadManager> thread_manager_;
    std::shared_ptr<MemoryPool> memory_pool_;
    
    // Results storage
    std::vector<CorrelationResult> last_results_;
    PerformanceMetrics performance_metrics_;
    
    // Helper methods
    cv::Mat preprocessImage(const cv::Mat& input);
    void validateImages(const cv::Mat& image_a, const cv::Mat& image_b);
    void updatePerformanceMetrics(const CorrelationResult& result);
};

} // namespace imgreg 