#include "fft_correlator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace imgreg {

FFTCorrelator::FFTCorrelator()
    : forward_plan_(nullptr),
      inverse_plan_(nullptr),
      input_buffer_(nullptr),
      fft_buffer_(nullptr),
      output_buffer_(nullptr),
      width_(0),
      height_(0),
      thread_count_(1),
      plan_type_(FFTW_ESTIMATE),
      optimizations_enabled_(true),
      precision_(1e-6),
      last_execution_time_(std::chrono::microseconds(0)),
      memory_usage_(0.0),
      memory_pool_(nullptr),
      aligned_buffer_(nullptr) {
    
    // Initialize FFTW threading if available
    #ifdef FFTW_THREADS_AVAILABLE
    fftw_init_threads();
    fftw_plan_with_nthreads(thread_count_);
    #endif
}

FFTCorrelator::~FFTCorrelator() {
    cleanupFFTW();
    #ifdef FFTW_THREADS_AVAILABLE
    fftw_cleanup_threads();
    #endif
}

CorrelationResult FFTCorrelator::correlate(const cv::Mat& image1, const cv::Mat& image2) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    validateInputs(image1, image2);
    
    // Initialize FFTW if needed
    if (width_ != image1.cols || height_ != image1.rows) {
        initializeFFTW(image1.cols, image1.rows);
    }
    
    // Perform FFT on both images
    auto fft_a = performFFT(image1);
    auto fft_b = performFFT(image2);
    
    // Multiply complex conjugate
    std::vector<std::vector<std::complex<double>>> correlation_fft(height_, std::vector<std::complex<double>>(width_));
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            correlation_fft[i][j] = fft_a[i][j] * std::conj(fft_b[i][j]);
        }
    }
    
    // Perform inverse FFT
    cv::Mat correlation_result = performIFFT(correlation_fft);
    
    // Find peak location
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(correlation_result, &min_val, &max_val, &min_loc, &max_loc);
    
    // Handle autocorrelation peak at (0,0)
    // The peak at (0,0) may be the autocorrelation peak rather than cross-correlation peak
    int offset_x = max_loc.x;
    int offset_y = max_loc.y;
    
    // Check for second peak if first peak is at (0,0)
    // For zero shifts, the peak at (0,0) is correct
    if (max_loc.x == 0 && max_loc.y == 0) {
        // Check correlation value to determine if this is a zero shift
        double correlation_threshold = 0.9 * max_val; // 90% of max value
        
        // Search for other peaks that may indicate a non-zero shift
        cv::Mat mask = cv::Mat::ones(correlation_result.size(), CV_8U);
        int exclude_radius = 2; // Exclude 5x5 region around origin
        for (int i = -exclude_radius; i <= exclude_radius; i++) {
            for (int j = -exclude_radius; j <= exclude_radius; j++) {
                int x = (i + width_) % width_;
                int y = (j + height_) % height_;
                if (x < 0) x += width_;
                if (y < 0) y += height_;
                mask.at<uchar>(y, x) = 0;
            }
        }
        
        // Find maximum excluding the origin region
        cv::Point second_peak_loc;
        double second_peak_val;
        cv::minMaxLoc(correlation_result, &min_val, &second_peak_val, &min_loc, &second_peak_loc, mask);
        
        // Use second peak if it is sufficiently strong
        if (second_peak_val > correlation_threshold) {
            offset_x = second_peak_loc.x;
            offset_y = second_peak_loc.y;
        }
        // Otherwise, keep the (0,0) peak as zero shift
    }
    
    // Handle wrap-around for shifts larger than half the image size
    if (offset_x > width_ / 2) {
        offset_x -= width_;
    }
    if (offset_y > height_ / 2) {
        offset_y -= height_;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    last_execution_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Create result
    CorrelationResult result;
    result.correlation_value = max_val;
    result.offset_x = offset_x;
    result.offset_y = offset_y;
    result.execution_time = last_execution_time_;
    result.method_name = "FFT";
    result.frequency_domain = fft_a; // Store frequency domain data
    
    return result;
}

void FFTCorrelator::setThreadCount(int threads) {
    thread_count_ = threads;
    #ifdef FFTW_THREADS_AVAILABLE
    fftw_plan_with_nthreads(thread_count_);
    #endif
}

void FFTCorrelator::setMemoryPool(std::shared_ptr<MemoryPool> pool) {
    memory_pool_ = pool;
}

void FFTCorrelator::setFFTPlanType(unsigned plan_type) {
    plan_type_ = plan_type;
}

std::chrono::microseconds FFTCorrelator::getLastExecutionTime() const {
    return last_execution_time_;
}

double FFTCorrelator::getMemoryUsage() const {
    return memory_usage_;
}

void FFTCorrelator::enableOptimizations(bool enable) {
    optimizations_enabled_ = enable;
}

void FFTCorrelator::setPrecision(double precision) {
    precision_ = precision;
}

void FFTCorrelator::initializeFFTW(int width, int height) {
    cleanupFFTW();
    
    // Validate input dimensions
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid dimensions in initializeFFTW: width=" + std::to_string(width) + ", height=" + std::to_string(height));
    }
    
    width_ = width;
    height_ = height;
    
    std::cout << "Initializing FFTW with dimensions: " << width_ << "x" << height_ << std::endl;
    
    // Allocate FFTW buffers
    input_buffer_ = (double*)fftw_malloc(width * height * sizeof(double));
    fft_buffer_ = (fftw_complex*)fftw_malloc(width * height * sizeof(fftw_complex));
    output_buffer_ = (double*)fftw_malloc(width * height * sizeof(double));
    
    if (!input_buffer_ || !fft_buffer_ || !output_buffer_) {
        throw std::runtime_error("Failed to allocate FFTW buffers");
    }
    
    // Create FFTW plans
    forward_plan_ = fftw_plan_dft_2d(height, width, fft_buffer_, fft_buffer_, FFTW_FORWARD, plan_type_);
    inverse_plan_ = fftw_plan_dft_2d(height, width, fft_buffer_, fft_buffer_, FFTW_BACKWARD, plan_type_);
    
    if (!forward_plan_ || !inverse_plan_) {
        throw std::runtime_error("Failed to create FFTW plans");
    }
    
    std::cout << "FFTW initialization completed" << std::endl;
}

void FFTCorrelator::cleanupFFTW() {
    if (forward_plan_) {
        fftw_destroy_plan(forward_plan_);
        forward_plan_ = nullptr;
    }
    if (inverse_plan_) {
        fftw_destroy_plan(inverse_plan_);
        inverse_plan_ = nullptr;
    }
    if (input_buffer_) {
        fftw_free(input_buffer_);
        input_buffer_ = nullptr;
    }
    if (fft_buffer_) {
        fftw_free(fft_buffer_);
        fft_buffer_ = nullptr;
    }
    if (output_buffer_) {
        fftw_free(output_buffer_);
        output_buffer_ = nullptr;
    }
}

std::vector<std::vector<std::complex<double>>> FFTCorrelator::performFFT(const cv::Mat& image) {
    // Convert image to double precision
    cv::Mat double_image;
    image.convertTo(double_image, CV_64F);
    
    // Convert real data to complex format
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            fft_buffer_[i * width_ + j][0] = double_image.at<double>(i, j); // Real part
            fft_buffer_[i * width_ + j][1] = 0.0; // Imaginary part
        }
    }
    
    // Execute forward FFT
    fftw_execute(forward_plan_);
    
    // Convert to complex vector
    std::vector<std::vector<std::complex<double>>> result(height_, std::vector<std::complex<double>>(width_));
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            result[i][j] = std::complex<double>(fft_buffer_[i * width_ + j][0], 
                                               fft_buffer_[i * width_ + j][1]);
        }
    }
    
    return result;
}

cv::Mat FFTCorrelator::performIFFT(const std::vector<std::vector<std::complex<double>>>& fft_data) {
    // Validate dimensions
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Invalid dimensions in performIFFT: width=" + std::to_string(width_) + ", height=" + std::to_string(height_));
    }
    
    if (fft_data.size() != static_cast<size_t>(height_) || fft_data[0].size() != static_cast<size_t>(width_)) {
        throw std::runtime_error("FFT data dimensions don't match: expected " + std::to_string(width_) + "x" + std::to_string(height_) + 
                                ", got " + std::to_string(fft_data[0].size()) + "x" + std::to_string(fft_data.size()));
    }
    
    // Copy complex data to FFT buffer
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            fft_buffer_[i * width_ + j][0] = fft_data[i][j].real();
            fft_buffer_[i * width_ + j][1] = fft_data[i][j].imag();
        }
    }
    
    // Execute inverse FFT
    fftw_execute(inverse_plan_);
    
    // Create OpenCV Mat from FFT buffer
    cv::Mat result(height_, width_, CV_64F);
    
    // Copy data and normalize
    // FFTW does not normalize automatically
    double normalization_factor = 1.0 / (width_ * height_);
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            result.at<double>(i, j) = fft_buffer_[i * width_ + j][0] * normalization_factor;
        }
    }
    
    return result;
}

void FFTCorrelator::optimizeMemoryLayout() {
    // Memory optimization placeholder
}

void FFTCorrelator::validateInputs(const cv::Mat& image1, const cv::Mat& image2) {
    if (image1.empty() || image2.empty()) {
        throw std::invalid_argument("Input images cannot be empty");
    }
    if (image1.size() != image2.size()) {
        throw std::invalid_argument("Input images must have the same size");
    }
    if (image1.type() != image2.type()) {
        throw std::invalid_argument("Input images must have the same type");
    }
}

} // namespace imgreg 