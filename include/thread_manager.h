#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <memory>

namespace imgreg {

class ThreadManager {
public:
    explicit ThreadManager(int thread_count = std::thread::hardware_concurrency());
    ~ThreadManager();
    
    // Task submission
    void submitTask(std::function<void()> task);
    void submitTasks(const std::vector<std::function<void()>>& tasks);
    
    // Thread control
    void start();
    void stop();
    void waitForCompletion();
    
    // Configuration
    void setThreadCount(int thread_count);
    int getThreadCount() const;
    int getActiveThreadCount() const;
    
    // Performance monitoring
    double getThreadUtilization() const;
    double getSpeedupRatio() const;
    void resetMetrics();

private:
    // Thread worker function
    void workerThread();
    
    // Task management
    bool getNextTask(std::function<void()>& task);
    
    // Thread pool state
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> task_queue_;
    
    // Synchronization
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    
    // Control flags
    std::atomic<bool> stop_flag_;
    std::atomic<bool> started_;
    
    // Configuration
    int thread_count_;
    
    // Performance metrics
    std::atomic<int> active_threads_;
    std::atomic<int> completed_tasks_;
    std::atomic<int> total_tasks_;
    mutable std::mutex metrics_mutex_;
    
    // Timing
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

// Utility class for parallel execution
class ParallelExecutor {
public:
    explicit ParallelExecutor(int thread_count = std::thread::hardware_concurrency());
    
    // Parallel for loop
    template<typename Iterator, typename Function>
    void parallelFor(Iterator begin, Iterator end, Function func);
    
    // Parallel reduction
    template<typename Iterator, typename T, typename BinaryOp>
    T parallelReduce(Iterator begin, Iterator end, T init, BinaryOp op);
    
    // Chunk-based processing
    template<typename Function>
    void processChunks(int total_size, int chunk_size, Function func);

private:
    std::shared_ptr<ThreadManager> thread_manager_;
};

} // namespace imgreg 