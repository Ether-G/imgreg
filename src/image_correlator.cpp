#include "image_correlator.h"
#include "spatial_correlator.h"
#include "fft_correlator.h"
#include "pearson_correlator.h"
#include "performance_analyzer.h"
#include "thread_manager.h"
#include "memory_pool.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace imgreg {

ImageCorrelator::ImageCorrelator() 
    : thread_count_(std::thread::hardware_concurrency()),
      image_width_(1024),
      image_height_(1024),
      performance_profiling_enabled_(false) {
    
    // Initialize components
    spatial_correlator_ = std::make_unique<SpatialCorrelator>();
    fft_correlator_ = std::make_unique<FFTCorrelator>();
    pearson_correlator_ = std::make_unique<PearsonCorrelator>();
    performance_analyzer_ = std::make_unique<PerformanceAnalyzer>();
    thread_manager_ = std::make_unique<ThreadManager>(thread_count_);
    memory_pool_ = std::make_shared<MemoryPool>();
    
    // Configure components
    spatial_correlator_->setThreadCount(thread_count_);
    fft_correlator_->setThreadCount(thread_count_);
    pearson_correlator_->setThreadCount(thread_count_);
    
    // Set memory pools
    auto pool_ptr = memory_pool_;
    spatial_correlator_->setMemoryPool(pool_ptr);
    fft_correlator_->setMemoryPool(pool_ptr);
    pearson_correlator_->setMemoryPool(pool_ptr);
}

ImageCorrelator::~ImageCorrelator() = default;

CorrelationResult ImageCorrelator::correlate(const cv::Mat& image_a, const cv::Mat& image_b, 
                                            const std::string& method) {
    // Validate input images
    validateImages(image_a, image_b);
    
    // Preprocess images
    cv::Mat processed_a = preprocessImage(image_a);
    cv::Mat processed_b = preprocessImage(image_b);
    
    CorrelationResult result;
    
    if (method == "spatial") {
        result = spatialCorrelate(processed_a, processed_b);
    } else if (method == "fft") {
        result = fftCorrelate(processed_a, processed_b);
    } else if (method == "pearson") {
        result = pearsonCorrelate(processed_a, processed_b);
    } else if (method == "all") {
        // Run all methods and return the best result
        auto spatial_result = spatialCorrelate(processed_a, processed_b);
        auto fft_result = fftCorrelate(processed_a, processed_b);
        auto pearson_result = pearsonCorrelate(processed_a, processed_b);
        
        // Choose the result with the highest correlation value
        if (spatial_result.correlation_value >= fft_result.correlation_value && 
            spatial_result.correlation_value >= pearson_result.correlation_value) {
            result = spatial_result;
        } else if (fft_result.correlation_value >= pearson_result.correlation_value) {
            result = fft_result;
        } else {
            result = pearson_result;
        }
    }
    
    if (performance_profiling_enabled_) {
        updatePerformanceMetrics(result);
    }
    
    return result;
}

CorrelationResult ImageCorrelator::spatialCorrelate(const cv::Mat& image_a, const cv::Mat& image_b) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CorrelationResult result = spatial_correlator_->correlate(image_a, image_b);
    result.method_name = "spatial";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return result;
}

CorrelationResult ImageCorrelator::fftCorrelate(const cv::Mat& image_a, const cv::Mat& image_b) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CorrelationResult result = fft_correlator_->correlate(image_a, image_b);
    result.method_name = "fft";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return result;
}

CorrelationResult ImageCorrelator::pearsonCorrelate(const cv::Mat& image_a, const cv::Mat& image_b) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CorrelationResult result = pearson_correlator_->correlate(image_a, image_b);
    result.method_name = "pearson";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return result;
}

std::vector<CorrelationResult> ImageCorrelator::benchmarkAllMethods(const cv::Mat& image_a, 
                                                                   const cv::Mat& image_b,
                                                                   int num_threads) {
    std::vector<CorrelationResult> results;
    
    // Set thread count for this benchmark
    int original_threads = thread_count_;
    setThreadCount(num_threads);
    
    // Run all methods
    results.push_back(spatialCorrelate(image_a, image_b));
    results.push_back(fftCorrelate(image_a, image_b));
    results.push_back(pearsonCorrelate(image_a, image_b));
    
    // Restore original thread count
    setThreadCount(original_threads);
    
    return results;
}

void ImageCorrelator::setThreadCount(int thread_count) {
    thread_count_ = thread_count;
    spatial_correlator_->setThreadCount(thread_count);
    fft_correlator_->setThreadCount(thread_count);
    pearson_correlator_->setThreadCount(thread_count);
    thread_manager_->setThreadCount(thread_count);
}

void ImageCorrelator::setImageSize(int width, int height) {
    image_width_ = width;
    image_height_ = height;
    
}

void ImageCorrelator::enablePerformanceProfiling(bool enable) {
    performance_profiling_enabled_ = enable;
    performance_analyzer_->enableDetailedLogging(enable);
}

PerformanceMetrics ImageCorrelator::getPerformanceMetrics() const {
    return performance_analyzer_->analyzePerformance();
}

void ImageCorrelator::exportResults(const std::string& filename) const {
    performance_analyzer_->exportToCSV(filename + ".csv");
    performance_analyzer_->exportToJSON(filename + ".json");
}

void ImageCorrelator::generateReport(const std::string& filename) const {
    std::ofstream report(filename);
    report << performance_analyzer_->generateReport();
    report.close();
}

cv::Mat ImageCorrelator::preprocessImage(const cv::Mat& input) {
    cv::Mat processed;
    
    // Convert to grayscale if needed
    if (input.channels() > 1) {
        cv::cvtColor(input, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = input.clone();
    }
    
    // Convert to double precision for better accuracy
    processed.convertTo(processed, CV_64F);
    
    // Normalize to [0, 1] range
    cv::normalize(processed, processed, 0.0, 1.0, cv::NORM_MINMAX);
    
    return processed;
}

void ImageCorrelator::validateImages(const cv::Mat& image_a, const cv::Mat& image_b) {
    if (image_a.empty() || image_b.empty()) {
        throw std::invalid_argument("One or both input images are empty");
    }
    
    if (image_a.size() != image_b.size()) {
        throw std::invalid_argument("Image dimensions must match");
    }
    
    if (image_a.type() != image_b.type()) {
        throw std::invalid_argument("Image types must match");
    }
}

void ImageCorrelator::updatePerformanceMetrics(const CorrelationResult& result) {
    performance_analyzer_->recordResult(result);
    
    // Note: Cannot modify const reference, so we skip updating cpu_utilization
    // The performance analyzer will handle this internally
}

} // namespace imgreg 