#pragma once

#include "correlation_result.h"
#include "memory_pool.h"
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>

namespace imgreg {

class FFTCorrelator {
public:
    FFTCorrelator();
    ~FFTCorrelator();
    
    CorrelationResult correlate(const cv::Mat& image1, const cv::Mat& image2);
    
    // Configuration methods
    void setThreadCount(int threads);
    void setMemoryPool(std::shared_ptr<MemoryPool> pool);
    void setFFTPlanType(unsigned plan_type);
    
    // Performance methods
    std::chrono::microseconds getLastExecutionTime() const;
    double getMemoryUsage() const;
    
    // Optimization methods
    void enableOptimizations(bool enable);
    void setPrecision(double precision);
    
private:
    // FFTW plans
    fftw_plan forward_plan_;
    fftw_plan inverse_plan_;
    
    // Data buffers
    double* input_buffer_;
    fftw_complex* fft_buffer_;
    double* output_buffer_;
    
    // Configuration
    int width_;
    int height_;
    int thread_count_;
    unsigned plan_type_;
    bool optimizations_enabled_;
    double precision_;
    
    // Performance tracking
    std::chrono::microseconds last_execution_time_;
    double memory_usage_;
    
    // Memory management
    std::shared_ptr<MemoryPool> memory_pool_;
    void* aligned_buffer_;
    
    // Helper methods
    void initializeFFTW(int width, int height);
    void cleanupFFTW();
    std::vector<std::vector<std::complex<double>>> performFFT(const cv::Mat& image);
    cv::Mat performIFFT(const std::vector<std::vector<std::complex<double>>>& fft_data);
    void optimizeMemoryLayout();
    void validateInputs(const cv::Mat& image1, const cv::Mat& image2);
};

} // namespace imgreg 