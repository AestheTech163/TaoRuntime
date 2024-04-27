#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <future>
#include <condition_variable>
#include <queue>


namespace concur {

typedef std::function<void(void)> TaskType;
typedef std::function<void(uint64_t, uint64_t)> TaskType_For;

class ThreadPool {

public:
    std::vector<std::thread> threads_;
    std::queue<TaskType> tasks_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_wait_;

    bool is_closed_;
    std::atomic<uint64_t> active_num_;
    uint64_t num_tasks_done_;
    uint64_t num_threads_needed_;
    const uint64_t pool_capacity_;

public:

    ThreadPool(uint64_t capacity): 
        pool_capacity_(capacity),
        is_closed_(false),
        num_tasks_done_(0),
        num_threads_needed_(0),
        active_num_(0) {

        auto thread_main_loop = [this]() {

            TaskType task;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    auto pred = [this]() { return is_closed_ || !tasks_.empty(); };
                    cv_.wait(lock, pred);

                    if (is_closed_ && tasks_.empty())  // 处于正在关闭状态，且任务队列为空，则关闭线程池
                        return;

                    task = std::move(tasks_.front());
                    tasks_.pop();

                    active_num_++;
                    // std::cout << "fetceh task" << std::endl;
                }

                task();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    active_num_--;
                    num_tasks_done_++;
                    num_threads_needed_--;
                    // std::cout << "num_threads_needed_: " << num_threads_needed_ << std::endl;
                    if (num_threads_needed_ == 0) {  // no more works 
                        cv_wait_.notify_one();
                    }
                }
                // 到这里，我们已经完成了一个任务，继续回到 wait 步，等待下一个任务
            }   // while (true)
        }; 

        for (int i=0; i<pool_capacity_; i++)
            threads_.emplace_back(thread_main_loop);
    }

    // void setNumWorksTodo(uint64_t num) { num_tasks_todo_ = num; }

    void resetStat() { num_tasks_done_ = 0; }

    uint64_t numTasksDone() const { return num_tasks_done_; }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_closed_ = true;
        }
        cv_.notify_all();
        for (auto& th : threads_)
            th.join();
    }

    bool isOpen() const { return !is_closed_; }

    bool isRunning() const { return !is_closed_ && (!tasks_.empty() || active_num_>0); }

    template <typename FuncType, typename ... ArgsType>
    void addTask(FuncType&& func, ArgsType&& ...args) {
        if (is_closed_)
            throw std::runtime_error("Can not add task to closed thread pool.");

        if (active_num_ < pool_capacity_) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto task_bind = std::bind(func, args...);
            tasks_.emplace(task_bind);
            cv_.notify_one();
        } else {
            std::cout << "herer" << std::endl;
            func(args...);
        }
    }

    void wait_task_done() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto predicate = [this] () -> bool {
            // std::cout << "from Main thread => num_threads_needed_: " << num_threads_needed_ << std::endl;
            return (num_threads_needed_==0);
        };
        cv_wait_.wait(lock, predicate);
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_closed_ = true;
        cv_.notify_all();
    }

    void parallelFor(std::ptrdiff_t num_works, const TaskType_For& func) {
        if (num_works <= 0) return;

        uint64_t num_threads_needed = std::min((uint64_t)num_works, pool_capacity_);

        std::vector<uint64_t> workload_dispatch(num_threads_needed, num_works/num_threads_needed);
        for (int i=0; i<num_works%num_threads_needed; i++)
            workload_dispatch[i]++;

        num_threads_needed_ = num_threads_needed;
        uint64_t begin = 0, end;
        for (int i=0; i<num_threads_needed; i++) {
            // func(block_size*i, block_size*i+block_size);
            end = begin + workload_dispatch[i];
            addTask(func, begin, end);
            begin = end;
        }
        wait_task_done();
    }
};

} // namespace concur

