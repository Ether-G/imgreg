#include "image_correlator.h"
#include "performance_analyzer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <filesystem>

using namespace imgreg;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "Options:\n"
              << "  -h, --help              Show this help message\n"
              << "  -i, --image-a <file>    First image file\n"
              << "  -j, --image-b <file>    Second image file\n"
              << "  -m, --method <method>   Correlation method (spatial, fft, pearson, all)\n"
              << "  -t, --threads <num>     Number of threads (default: auto-detect)\n"
              << "  -s, --size <size>       Image size for synthetic tests (256, 1024, 4096)\n"
              << "  -b, --benchmark         Run full benchmark suite\n"
              << "  -o, --output <dir>      Output directory for results\n"
              << "  -v, --verbose           Enable verbose output\n"
              << "  --synthetic             Generate synthetic test images\n"
              << "  --profile               Enable performance profiling\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " -i image1.jpg -j image2.jpg -m all\n"
              << "  " << program_name << " -b -s 1024 -t 8 -o results/\n"
              << "  " << program_name << " --synthetic --benchmark -s 4096\n";
}

cv::Mat generateSyntheticImage(int width, int height, int pattern_type = 0) {
    cv::Mat image(height, width, CV_8UC1);
    
    switch (pattern_type) {
        case 0: // Random noise
            cv::randu(image, 0, 255);
            break;
        case 1: // Checkerboard
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    image.at<uchar>(y, x) = ((x / 32) + (y / 32)) % 2 * 255;
                }
            }
            break;
        case 2: // Gradient
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    image.at<uchar>(y, x) = (x + y) % 256;
                }
            }
            break;
        case 3: // Sine wave pattern
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    double value = 127.5 + 127.5 * sin(2 * M_PI * x / 64.0) * cos(2 * M_PI * y / 64.0);
                    image.at<uchar>(y, x) = static_cast<uchar>(value);
                }
            }
            break;
    }
    
    return image;
}

void runSingleTest(const std::string& image_a_path, const std::string& image_b_path, 
                   const std::string& method, int thread_count, bool verbose) {
    std::cout << "Loading images..." << std::endl;
    
    cv::Mat image_a = cv::imread(image_a_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_b = cv::imread(image_b_path, cv::IMREAD_GRAYSCALE);
    
    if (image_a.empty() || image_b.empty()) {
        std::cerr << "Error: Could not load one or both images." << std::endl;
        return;
    }
    
    if (verbose) {
        std::cout << "Image A: " << image_a.cols << "x" << image_a.rows << std::endl;
        std::cout << "Image B: " << image_b.cols << "x" << image_b.rows << std::endl;
    }
    
    ImageCorrelator correlator;
    correlator.setThreadCount(thread_count);
    correlator.enablePerformanceProfiling(true);
    
    std::cout << "Running correlation with method: " << method << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CorrelationResult result = correlator.correlate(image_a, image_b, method);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "\n=== Correlation Results ===" << std::endl;
    std::cout << "Method: " << result.method_name << std::endl;
    std::cout << "Correlation Value: " << result.correlation_value << std::endl;
    std::cout << "Offset: (" << result.offset_x << ", " << result.offset_y << ")" << std::endl;
    std::cout << "Execution Time: " << duration.count() << " μs" << std::endl;
    std::cout << "Memory Usage: " << result.memory_usage_bytes << " bytes" << std::endl;
    std::cout << "Threads Used: " << result.thread_count << std::endl;
    
    if (method == "pearson") {
        std::cout << "Pearson Coefficient: " << result.pearson_coefficient << std::endl;
    }
}

void runBenchmarkSuite(int image_size, int max_threads, const std::string& output_dir, bool verbose) {
    std::cout << "Running benchmark suite with " << image_size << "x" << image_size << " images..." << std::endl;
    
    // Generate test images
    cv::Mat base_image = generateSyntheticImage(image_size, image_size, 3); // Sine wave pattern
    cv::Mat shifted_image = generateSyntheticImage(image_size, image_size, 3);
    
    // Apply a known shift for validation
    int known_shift_x = 10;
    int known_shift_y = 15;
    cv::Mat translation_matrix = cv::getRotationMatrix2D(cv::Point2f(0, 0), 0, 1.0);
    translation_matrix.at<double>(0, 2) = known_shift_x;
    translation_matrix.at<double>(1, 2) = known_shift_y;
    cv::warpAffine(base_image, shifted_image, translation_matrix, shifted_image.size());
    
    ImageCorrelator correlator;
    PerformanceAnalyzer analyzer;
    
    std::vector<std::string> methods = {"spatial", "fft", "pearson"};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    if (max_threads > 0) {
        thread_counts.clear();
        for (int t = 1; t <= max_threads; t *= 2) {
            thread_counts.push_back(t);
        }
    }
    
    analyzer.startSession();
    
    for (const auto& method : methods) {
        std::cout << "\n=== Testing " << method << " method ===" << std::endl;
        
        for (int threads : thread_counts) {
            if (threads > std::thread::hardware_concurrency()) {
                continue;
            }
            
            correlator.setThreadCount(threads);
            
            if (verbose) {
                std::cout << "  Testing with " << threads << " threads..." << std::endl;
            }
            
            auto start_time = std::chrono::high_resolution_clock::now();
            CorrelationResult result = correlator.correlate(base_image, shifted_image, method);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            result.execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            analyzer.recordResult(result);
            
            if (verbose) {
                std::cout << "    Time: " << result.execution_time.count() << " μs" << std::endl;
                std::cout << "    Memory: " << result.memory_usage_bytes << " bytes" << std::endl;
                std::cout << "    Correlation: " << result.correlation_value << std::endl;
                std::cout << "    Offset: (" << result.offset_x << ", " << result.offset_y << ")" << std::endl;
                std::cout << "    Expected: (" << known_shift_x << ", " << known_shift_y << ")" << std::endl;
            }
        }
    }
    
    analyzer.endSession();
    
    // Generate reports
    std::cout << "\n=== Generating Reports ===" << std::endl;
    
    if (!output_dir.empty()) {
        std::string csv_file = output_dir + "/benchmark_results.csv";
        std::string json_file = output_dir + "/benchmark_results.json";
        std::string report_file = output_dir + "/benchmark_report.txt";
        
        analyzer.exportToCSV(csv_file);
        analyzer.exportToJSON(json_file);
        
        std::ofstream report(report_file);
        report << analyzer.generateReport();
        report.close();
        
        std::cout << "Results saved to: " << output_dir << std::endl;
    }
    
    // Print summary
    std::cout << "\n=== Benchmark Summary ===" << std::endl;
    std::cout << analyzer.generateReport() << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_a_path, image_b_path, method = "all", output_dir = "results";
    int thread_count = std::thread::hardware_concurrency();
    int image_size = 1024;
    bool benchmark_mode = false;
    bool synthetic_mode = false;
    bool verbose = false;
    bool profile = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--image-a") {
            if (++i < argc) image_a_path = argv[i];
        } else if (arg == "-j" || arg == "--image-b") {
            if (++i < argc) image_b_path = argv[i];
        } else if (arg == "-m" || arg == "--method") {
            if (++i < argc) method = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) thread_count = std::atoi(argv[i]);
        } else if (arg == "-s" || arg == "--size") {
            if (++i < argc) image_size = std::atoi(argv[i]);
        } else if (arg == "-b" || arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) output_dir = argv[i];
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--synthetic") {
            synthetic_mode = true;
        } else if (arg == "--profile") {
            profile = true;
        }
    }
    
    // Validate arguments
    if (!benchmark_mode && (image_a_path.empty() || image_b_path.empty())) {
        std::cerr << "Error: Must provide both image files or use --benchmark mode." << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    if (method != "spatial" && method != "fft" && method != "pearson" && method != "all") {
        std::cerr << "Error: Invalid method. Use 'spatial', 'fft', 'pearson', or 'all'." << std::endl;
        return 1;
    }
    
    // Create output directory
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }
    
    try {
        if (benchmark_mode) {
            runBenchmarkSuite(image_size, thread_count, output_dir, verbose);
        } else {
            runSingleTest(image_a_path, image_b_path, method, thread_count, verbose);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 