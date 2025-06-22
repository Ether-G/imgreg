#include "thread_manager.h"
#include <iostream>
#include <algorithm>

namespace imgreg {

ThreadManager::ThreadManager(int thread_count)
    : thread_count_(thread_count),
      stop_flag_(false),
      started_(false),
      active_threads_(0),
      completed_tasks_(0),
      total_tasks_(0) {
    
    if (thread_count_ <= 0) {
        thread_count_ = std::thread::hardware_concurrency();
    }
}

ThreadManager::~ThreadManager() {
    stop();
    waitForCompletion();
}

void ThreadManager::submitTask(std::function<void()> task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    task_queue_.push(std::move(task));
    total_tasks_++;
    condition_.notify_one();
}

void ThreadManager::submitTasks(const std::vector<std::function<void()>>& tasks) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (const auto& task : tasks) {
        task_queue_.push(task);
        total_tasks_++;
    }
    condition_.notify_all();
}

void ThreadManager::start() {
    if (started_) {
        return;
    }
    
    started_ = true;
    stop_flag_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // Create worker threads
    for (int i = 0; i < thread_count_; ++i) {
        threads_.emplace_back(&ThreadManager::workerThread, this);
    }
}

void ThreadManager::stop() {
    stop_flag_ = true;
    condition_.notify_all();
}

void ThreadManager::waitForCompletion() {
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
    started_ = false;
    end_time_ = std::chrono::high_resolution_clock::now();
}

void ThreadManager::setThreadCount(int thread_count) {
    if (thread_count <= 0) {
        thread_count = std::thread::hardware_concurrency();
    }
    
    if (thread_count != thread_count_) {
        bool was_started = started_;
        if (was_started) {
            stop();
            waitForCompletion();
        }
        
        thread_count_ = thread_count;
        
        if (was_started) {
            start();
        }
    }
}

int ThreadManager::getThreadCount() const {
    return thread_count_;
}

int ThreadManager::getActiveThreadCount() const {
    return active_threads_.load();
}

double ThreadManager::getThreadUtilization() const {
    if (total_tasks_ == 0) {
        return 0.0;
    }
    
    return static_cast<double>(completed_tasks_) / total_tasks_;
}

double ThreadManager::getSpeedupRatio() const {
    if (thread_count_ <= 1) {
        return 1.0;
    }
    
    // Simple speedup calculation based on thread utilization
    return getThreadUtilization() * thread_count_;
}

void ThreadManager::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    completed_tasks_ = 0;
    total_tasks_ = 0;
    active_threads_ = 0;
}

void ThreadManager::workerThread() {
    while (!stop_flag_) {
        std::function<void()> task;
        
        if (getNextTask(task)) {
            active_threads_++;
            try {
                task();
                completed_tasks_++;
            } catch (const std::exception& e) {
                std::cerr << "Task execution failed: " << e.what() << std::endl;
            }
            active_threads_--;
        }
    }
}

bool ThreadManager::getNextTask(std::function<void()>& task) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    condition_.wait(lock, [this] {
        return !task_queue_.empty() || stop_flag_;
    });
    
    if (stop_flag_ && task_queue_.empty()) {
        return false;
    }
    
    if (!task_queue_.empty()) {
        task = std::move(task_queue_.front());
        task_queue_.pop();
        return true;
    }
    
    return false;
}

// ParallelExecutor implementation
ParallelExecutor::ParallelExecutor(int thread_count)
    : thread_manager_(std::make_shared<ThreadManager>(thread_count)) {
}

} // namespace imgreg 