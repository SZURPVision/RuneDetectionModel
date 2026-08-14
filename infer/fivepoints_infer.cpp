/**
 * FivePoints OpenVINO C++ 推理与性能评估
 * 支持图片目录与视频批量推理、性能统计与可视化
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// 默认配置参数
const int DEFAULT_KPT_DIM = 3;
const float DEFAULT_CONF_THRESHOLD = 0.8f;
const float DEFAULT_KPT_CONF_THRESHOLD = 0.8f;
const float DEFAULT_NMS_DIST_THRESHOLD = 30.0f;
const int DEFAULT_MIN_VALID_KPTS = 3;

const std::vector<cv::Scalar> KPT_COLORS = {
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 0, 255)};

const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
const std::vector<std::string> VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"};

struct Detection
{
    int class_id = -1;
    float confidence = 0.0f;
    float quality = 0.0f;
    cv::Point2f center{0.0f, 0.0f};
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kpt_confs;
};

struct TimingRecord
{
    std::string name;
    int detections = 0;
    float preprocess_time = 0.0f;
    float infer_time = 0.0f;
    float postprocess_time = 0.0f;
    float draw_time = 0.0f;
    float save_time = 0.0f;
    float total_time = 0.0f;
};

struct PerformanceStats
{
    int total_images = 0;
    int total_detections = 0;
    int warmup_iterations = 0;
    float warmup_total_time_ms = 0.0f;
    std::vector<TimingRecord> records;

    void add_record(const TimingRecord &rec)
    {
        records.push_back(rec);
        total_images++;
        total_detections += rec.detections;
    }

    template <typename F>
    std::vector<float> extract(F &&getter) const
    {
        std::vector<float> res;
        res.reserve(records.size());
        for (const auto &r : records)
            res.push_back(getter(r));
        return res;
    }

    static float mean(const std::vector<float> &v)
    {
        return v.empty() ? 0.0f : std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    }

    static float stddev(const std::vector<float> &v)
    {
        if (v.size() < 2)
            return 0.0f;
        float m = mean(v);
        float sum = 0.0f;
        for (float x : v)
            sum += (x - m) * (x - m);
        return std::sqrt(sum / v.size());
    }

    static float min_val(const std::vector<float> &v)
    {
        return v.empty() ? 0.0f : *std::min_element(v.begin(), v.end());
    }

    static float max_val(const std::vector<float> &v)
    {
        return v.empty() ? 0.0f : *std::max_element(v.begin(), v.end());
    }

    static float percentile(std::vector<float> v, float p)
    {
        if (v.empty())
            return 0.0f;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>((p / 100.0f) * (v.size() - 1));
        return v[idx];
    }
};

class FivePointsDetector
{
  public:
    FivePointsDetector(const std::string &model_path, const std::string &device = "AUTO",
                       int num_classes_override = -1, int num_keypoints_override = -1,
                       float conf_threshold = DEFAULT_CONF_THRESHOLD,
                       float kpt_conf_threshold = DEFAULT_KPT_CONF_THRESHOLD,
                       float nms_dist_threshold = DEFAULT_NMS_DIST_THRESHOLD,
                       int min_valid_kpts = DEFAULT_MIN_VALID_KPTS)
        : conf_threshold_(conf_threshold),
          kpt_conf_threshold_(kpt_conf_threshold),
          nms_dist_threshold_(nms_dist_threshold),
          min_valid_kpts_(min_valid_kpts)
    {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(model_path);
        class_names_ = read_class_names_from_model(model);
        const auto model_input_shape = model->input().get_shape();

        if (model_input_shape.size() != 4)
        {
            throw std::runtime_error("模型输入非4维 (NCHW)");
        }
        input_height_ = static_cast<int>(model_input_shape[2]);
        input_width_ = static_cast<int>(model_input_shape[3]);

        // PrePostProcessor 配置 (BGR u8 NHWC -> RGB f32 [0,1] NCHW)
        auto ppp = ov::preprocess::PrePostProcessor(model);
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0f);
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();

        compiled_model_ = core.compile_model(model, device,
                                             ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        infer_request_ = compiled_model_.create_infer_request();

        const auto output_shape = compiled_model_.output().get_shape();
        if (output_shape.size() != 3)
        {
            throw std::runtime_error("模型输出维度非3维 ([N,C,A] 或 [N,A,C])");
        }

        // 识别输出布局 [N,C,A] 或 [N,A,C]
        if (output_shape[1] <= 256 && output_shape[2] >= 100)
        {
            output_layout_nca_ = true;
            output_channels_ = static_cast<int>(output_shape[1]);
            num_anchors_ = static_cast<int>(output_shape[2]);
        }
        else
        {
            output_layout_nca_ = false;
            num_anchors_ = static_cast<int>(output_shape[1]);
            output_channels_ = static_cast<int>(output_shape[2]);
        }

        resolve_head_dims(num_classes_override, num_keypoints_override);
        ensure_class_names();

        std::cout << "[Info] 模型初始化成功 (Device: " << device << ")\n"
                  << "       输入尺寸: " << input_width_ << "x" << input_height_
                  << " | 布局: " << (output_layout_nca_ ? "[N,C,A]" : "[N,A,C]")
                  << " | 通道数: " << output_channels_
                  << " (类别: " << num_classes_ << ", 关键点: " << num_keypoints_ << ")\n";
    }

    void preprocess_letterbox(const cv::Mat &src, cv::Mat &dst, float &scale, int &pad_w, int &pad_h)
    {
        scale = std::min(static_cast<float>(input_width_) / src.cols,
                         static_cast<float>(input_height_) / src.rows);
        int new_w = static_cast<int>(std::round(src.cols * scale));
        int new_h = static_cast<int>(std::round(src.rows * scale));

        pad_w = (input_width_ - new_w) / 2;
        pad_h = (input_height_ - new_h) / 2;
        int right = input_width_ - new_w - pad_w;
        int bottom = input_height_ - new_h - pad_h;

        cv::Mat resized;
        if (new_w != src.cols || new_h != src.rows)
            cv::resize(src, resized, cv::Size(new_w, new_h));
        else
            resized = src;

        if (pad_w > 0 || pad_h > 0 || right > 0 || bottom > 0)
            cv::copyMakeBorder(resized, dst, pad_h, bottom, pad_w, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        else
            dst = resized;
    }

    std::vector<Detection> infer(const cv::Mat &image, TimingRecord &rec)
    {
        float scale = 1.0f;
        int pad_w = 0, pad_h = 0;
        cv::Mat preprocessed_img;

        // 1. 预处理
        auto t0 = std::chrono::high_resolution_clock::now();
        preprocess_letterbox(image, preprocessed_img, scale, pad_w, pad_h);
        ov::Tensor input_tensor(ov::element::u8,
                                {1, (size_t)input_height_, (size_t)input_width_, 3},
                                preprocessed_img.data);
        infer_request_.set_input_tensor(input_tensor);
        auto t1 = std::chrono::high_resolution_clock::now();
        rec.preprocess_time = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 2. 推理
        infer_request_.infer();
        auto t2 = std::chrono::high_resolution_clock::now();
        rec.infer_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // 3. 后处理
        auto results = postprocess(scale, pad_w, pad_h, image.cols, image.rows);
        auto t3 = std::chrono::high_resolution_clock::now();
        rec.postprocess_time = std::chrono::duration<float, std::milli>(t3 - t2).count();
        rec.detections = static_cast<int>(results.size());

        return results;
    }

    std::vector<Detection> postprocess(float scale, int pad_w, int pad_h, int orig_w, int orig_h)
    {
        std::vector<Detection> detections;
        detections.reserve(64);
        const float *output_data = infer_request_.get_output_tensor().data<float>();

        auto get_val = [&](int c, int a) -> float {
            return output_layout_nca_ ? output_data[c * num_anchors_ + a]
                                      : output_data[a * output_channels_ + c];
        };

        for (int i = 0; i < num_anchors_; ++i)
        {
            int best_cls = -1;
            float max_conf = 0.0f;
            for (int c = 0; c < num_classes_; ++c)
            {
                float score = get_val(c, i);
                if (score > max_conf)
                {
                    max_conf = score;
                    best_cls = c;
                }
            }

            if (best_cls < 0 || max_conf < conf_threshold_)
                continue;

            Detection det;
            det.class_id = best_cls;
            det.confidence = max_conf;
            det.keypoints.reserve(num_keypoints_);
            det.kpt_confs.reserve(num_keypoints_);

            int valid_kpts = 0;
            float kpt_conf_sum = 0.0f;
            cv::Point2f valid_center(0.0f, 0.0f);
            cv::Point2f all_center(0.0f, 0.0f);
            bool has_negative_kpt = false;

            for (int k = 0; k < num_keypoints_; ++k)
            {
                int base = num_classes_ + k * kpt_dim_;
                float x = (get_val(base, i) - pad_w) / scale;
                float y = (get_val(base + 1, i) - pad_h) / scale;
                float k_conf = get_val(base + 2, i);

                if (x < 0.0f || y < 0.0f)
                {
                    has_negative_kpt = true;
                    break;
                }

                x = std::clamp(x, 0.0f, static_cast<float>(orig_w - 1));
                y = std::clamp(y, 0.0f, static_cast<float>(orig_h - 1));

                cv::Point2f pt(x, y);
                det.keypoints.push_back(pt);
                det.kpt_confs.push_back(k_conf);
                all_center += pt;

                if (k_conf >= kpt_conf_threshold_)
                {
                    valid_kpts++;
                    kpt_conf_sum += k_conf;
                    valid_center += pt;
                }
            }

            if (has_negative_kpt || valid_kpts < std::min(min_valid_kpts_, num_keypoints_))
                continue;

            float mean_kpt_conf = valid_kpts > 0 ? (kpt_conf_sum / valid_kpts) : 0.0f;
            det.quality = det.confidence * mean_kpt_conf;
            det.center = (valid_kpts > 0) ? (valid_center * (1.0f / valid_kpts))
                                          : (all_center * (1.0f / num_keypoints_));

            detections.push_back(std::move(det));
        }

        return nms(detections);
    }

    std::vector<Detection> nms(std::vector<Detection> &detections)
    {
        if (detections.empty())
            return {};

        std::sort(detections.begin(), detections.end(),
                  [](const Detection &a, const Detection &b) { return a.quality > b.quality; });

        std::vector<Detection> result;
        std::vector<bool> suppressed(detections.size(), false);
        const float dist_thresh_sq = nms_dist_threshold_ * nms_dist_threshold_;

        for (size_t i = 0; i < detections.size(); ++i)
        {
            if (suppressed[i])
                continue;
            result.push_back(detections[i]);
            const auto &c_i = detections[i].center;

            for (size_t j = i + 1; j < detections.size(); ++j)
            {
                if (suppressed[j])
                    continue;
                float dx = c_i.x - detections[j].center.x;
                float dy = c_i.y - detections[j].center.y;
                if ((dx * dx + dy * dy) < dist_thresh_sq)
                    suppressed[j] = true;
            }
        }
        return result;
    }

    void draw(cv::Mat &image, const std::vector<Detection> &detections) const
    {
        for (const auto &det : detections)
        {
            for (size_t k = 0; k < det.keypoints.size(); ++k)
            {
                if (det.kpt_confs[k] >= kpt_conf_threshold_)
                {
                    cv::Scalar color = KPT_COLORS[k % KPT_COLORS.size()];
                    cv::circle(image, det.keypoints[k], 3, color, -1);
                    cv::circle(image, det.keypoints[k], 3, cv::Scalar(255, 255, 255), 1);
                }
            }

            std::string cls_name = (det.class_id >= 0 && det.class_id < (int)class_names_.size())
                                       ? class_names_[det.class_id]
                                       : ("class" + std::to_string(det.class_id));
            std::string label = cls_name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(image,
                          cv::Point(det.center.x - 2, det.center.y - text_size.height - 4),
                          cv::Point(det.center.x + text_size.width + 2, det.center.y + 2),
                          cv::Scalar(0, 0, 0), -1);
            cv::putText(image, label, cv::Point(det.center.x, det.center.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }

    float warmup(const cv::Mat &image, int iterations = 10)
    {
        if (iterations <= 0)
            return 0.0f;
        std::cout << "[Info] 预热 " << iterations << " 次...\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        TimingRecord rec;
        for (int i = 0; i < iterations; ++i)
            infer(image, rec);
        auto t2 = std::chrono::high_resolution_clock::now();
        float total_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();
        std::cout << "[Info] 预热完成, 平均耗时: " << (total_ms / iterations) << " ms\n";
        return total_ms;
    }

  private:
    std::vector<std::string> read_class_names_from_model(const std::shared_ptr<ov::Model> &model) const
    {
        try
        {
            if (model && model->has_rt_info("model_info", "labels"))
            {
                auto labels = model->get_rt_info<std::vector<std::string>>("model_info", "labels");
                for (auto &name : labels)
                    std::replace(name.begin(), name.end(), '_', ' ');
                return labels;
            }
        }
        catch (...)
        {
        }
        return {};
    }

    void resolve_head_dims(int num_classes_override, int num_keypoints_override)
    {
        if (num_classes_override > 0 && num_keypoints_override > 0)
        {
            if (num_classes_override + num_keypoints_override * kpt_dim_ != output_channels_)
                throw std::runtime_error("参数 --nc 与 --nk 组合与模型输出通道不匹配");
            num_classes_ = num_classes_override;
            num_keypoints_ = num_keypoints_override;
            return;
        }

        if (num_classes_override > 0)
        {
            int remain = output_channels_ - num_classes_override;
            if (remain <= 0 || remain % kpt_dim_ != 0)
                throw std::runtime_error("参数 --nc 与输出通道不匹配");
            num_classes_ = num_classes_override;
            num_keypoints_ = remain / kpt_dim_;
            return;
        }

        if (num_keypoints_override > 0)
        {
            int nc = output_channels_ - num_keypoints_override * kpt_dim_;
            if (nc <= 0)
                throw std::runtime_error("参数 --nk 与输出通道不匹配");
            num_classes_ = nc;
            num_keypoints_ = num_keypoints_override;
            return;
        }

        if (!class_names_.empty())
        {
            int nc = static_cast<int>(class_names_.size());
            int remain = output_channels_ - nc;
            if (remain > 0 && remain % kpt_dim_ == 0)
            {
                num_classes_ = nc;
                num_keypoints_ = remain / kpt_dim_;
                return;
            }
        }

        // 默认匹配能量机关 3 类别 + 5 关键点 (3 + 5*3 = 18 通道)
        if (output_channels_ == 18)
        {
            num_classes_ = 3;
            num_keypoints_ = 5;
            return;
        }

        throw std::runtime_error("无法自动推断类别和关键点数，请使用 --nc 或 --nk 参数指定");
    }

    void ensure_class_names()
    {
        if ((int)class_names_.size() == num_classes_)
            return;
        class_names_.clear();
        if (num_classes_ == 3)
        {
            class_names_ = {"inactive", "small_activated", "big_activated"};
        }
        else
        {
            class_names_.reserve(num_classes_);
            for (int i = 0; i < num_classes_; ++i)
                class_names_.push_back("class" + std::to_string(i));
        }
    }

    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    int input_width_ = 640;
    int input_height_ = 640;
    int output_channels_ = 0;
    int num_anchors_ = 0;
    int num_classes_ = 0;
    int num_keypoints_ = 0;
    int kpt_dim_ = DEFAULT_KPT_DIM;
    float conf_threshold_ = DEFAULT_CONF_THRESHOLD;
    float kpt_conf_threshold_ = DEFAULT_KPT_CONF_THRESHOLD;
    float nms_dist_threshold_ = DEFAULT_NMS_DIST_THRESHOLD;
    int min_valid_kpts_ = DEFAULT_MIN_VALID_KPTS;
    std::vector<std::string> class_names_;
    bool output_layout_nca_ = true;
};

// 辅助文件扫描
std::vector<fs::path> get_image_files(const std::string &dir_path)
{
    std::vector<fs::path> files;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end())
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool is_video_file(const fs::path &path)
{
    if (!fs::is_regular_file(path))
        return false;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(VIDEO_EXTS.begin(), VIDEO_EXTS.end(), ext) != VIDEO_EXTS.end();
}

// 保存报告
void save_reports(const std::string &output_dir, const PerformanceStats &stats,
                  const std::string &model_path, const std::string &input_path)
{
    fs::create_directories(output_dir);
    std::string csv_path = output_dir + "/performance_details.csv";
    std::string json_path = output_dir + "/performance_report.json";

    std::ofstream csv(csv_path);
    csv << "name,detections,preprocess_ms,infer_ms,postprocess_ms,draw_ms,save_ms,total_ms\n";
    for (const auto &r : stats.records)
    {
        csv << r.name << "," << r.detections << ","
            << r.preprocess_time << "," << r.infer_time << "," << r.postprocess_time << ","
            << r.draw_time << "," << r.save_time << "," << r.total_time << "\n";
    }

    auto prep_times = stats.extract([](const TimingRecord &r) { return r.preprocess_time; });
    auto infer_times = stats.extract([](const TimingRecord &r) { return r.infer_time; });
    auto post_times = stats.extract([](const TimingRecord &r) { return r.postprocess_time; });
    auto total_times = stats.extract([](const TimingRecord &r) { return r.total_time; });

    float avg_infer = PerformanceStats::mean(infer_times);
    float avg_total = PerformanceStats::mean(total_times);

    std::ofstream json(json_path);
    json << std::fixed << std::setprecision(3);
    json << "{\n"
         << "  \"model\": \"" << model_path << "\",\n"
         << "  \"input\": \"" << input_path << "\",\n"
         << "  \"total_images\": " << stats.total_images << ",\n"
         << "  \"total_detections\": " << stats.total_detections << ",\n"
         << "  \"fps_infer\": " << (avg_infer > 0 ? 1000.0f / avg_infer : 0.0f) << ",\n"
         << "  \"fps_end2end\": " << (avg_total > 0 ? 1000.0f / avg_total : 0.0f) << ",\n"
         << "  \"timing_mean_ms\": {\n"
         << "    \"preprocess\": " << PerformanceStats::mean(prep_times) << ",\n"
         << "    \"infer\": " << avg_infer << ",\n"
         << "    \"postprocess\": " << PerformanceStats::mean(post_times) << ",\n"
         << "    \"total\": " << avg_total << "\n"
         << "  }\n"
         << "}\n";

    std::cout << "[Info] 报告保存至: " << output_dir << "\n";
}

void print_summary(const PerformanceStats &stats)
{
    auto prep = stats.extract([](const TimingRecord &r) { return r.preprocess_time; });
    auto infer = stats.extract([](const TimingRecord &r) { return r.infer_time; });
    auto post = stats.extract([](const TimingRecord &r) { return r.postprocess_time; });
    auto total = stats.extract([](const TimingRecord &r) { return r.total_time; });

    std::cout << "\n================ 性能评估报告 ================\n"
              << "总处理帧/图数: " << stats.total_images
              << " | 检测总数: " << stats.total_detections << "\n";

    auto print_line = [](const char *name, const std::vector<float> &v) {
        std::cout << std::left << std::setw(12) << name
                  << " 平均: " << std::setw(7) << std::fixed << std::setprecision(2) << PerformanceStats::mean(v)
                  << "ms | 标准差: " << std::setw(6) << PerformanceStats::stddev(v)
                  << "ms | [Min/Max: " << PerformanceStats::min_val(v) << " / " << PerformanceStats::max_val(v) << " ms]\n";
    };

    print_line("预处理", prep);
    print_line("推理", infer);
    print_line("后处理", post);
    print_line("端到端总计", total);

    float avg_infer = PerformanceStats::mean(infer);
    float avg_total = PerformanceStats::mean(total);
    std::cout << "----------------------------------------------\n"
              << "纯推理 FPS: " << (avg_infer > 0 ? 1000.0f / avg_infer : 0.0f)
              << " | 端到端 FPS: " << (avg_total > 0 ? 1000.0f / avg_total : 0.0f) << "\n"
              << "推理延迟 (P50/P90/P99): "
              << PerformanceStats::percentile(infer, 50) << " / "
              << PerformanceStats::percentile(infer, 90) << " / "
              << PerformanceStats::percentile(infer, 99) << " ms\n"
              << "==============================================\n\n";
}

void print_usage(const char *prog)
{
    std::cout << "用法: " << prog << " <model.xml> <input_path> [output_path] [options]\n"
              << "选项:\n"
              << "  --device D   推理设备 (默认: AUTO, 可选: CPU, GPU)\n"
              << "  --warmup N   预热次数 (默认: 10)\n"
              << "  --nc N       类别数 (可选, 默认从模型元数据自动推断)\n"
              << "  --nk N       关键点个数 (可选)\n"
              << "  --conf T     置信度阈值 (默认: 0.8)\n"
              << "  --kconf T    关键点置信度阈值 (默认: 0.8)\n"
              << "  --nms-dist T NMS中心点抑制距离 (默认: 30)\n"
              << "  --min-kpts N 至少有效关键点数 (默认: 3)\n"
              << "  --no-save    不保存可视化结果\n"
              << "  --benchmark  基准测试模式 (同 --no-save)\n";
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return -1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = "";
    std::string device = "AUTO";
    int warmup_iterations = 10;
    int num_classes = -1, num_keypoints = -1;
    float conf = DEFAULT_CONF_THRESHOLD;
    float kconf = DEFAULT_KPT_CONF_THRESHOLD;
    float nms_dist = DEFAULT_NMS_DIST_THRESHOLD;
    int min_kpts = DEFAULT_MIN_VALID_KPTS;
    bool save_results = true;

    for (int i = 3; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup_iterations = std::stoi(argv[++i]);
        else if (arg == "--nc" && i + 1 < argc) num_classes = std::stoi(argv[++i]);
        else if (arg == "--nk" && i + 1 < argc) num_keypoints = std::stoi(argv[++i]);
        else if (arg == "--conf" && i + 1 < argc) conf = std::stof(argv[++i]);
        else if (arg == "--kconf" && i + 1 < argc) kconf = std::stof(argv[++i]);
        else if (arg == "--nms-dist" && i + 1 < argc) nms_dist = std::stof(argv[++i]);
        else if (arg == "--min-kpts" && i + 1 < argc) min_kpts = std::stoi(argv[++i]);
        else if (arg == "--no-save" || arg == "--benchmark") save_results = false;
        else if (output_path.empty() && arg[0] != '-') output_path = arg;
    }

    try
    {
        fs::path in_path(input_path);
        bool is_video = is_video_file(in_path);

        FivePointsDetector detector(model_path, device, num_classes, num_keypoints,
                                   conf, kconf, nms_dist, min_kpts);
        PerformanceStats stats;

        if (is_video)
        {
            cv::VideoCapture cap(input_path);
            if (!cap.isOpened())
            {
                std::cerr << "[Error] 无法打开视频: " << input_path << "\n";
                return -1;
            }

            std::string out_video_path = output_path.empty()
                                             ? (in_path.parent_path() / (in_path.stem().string() + "_result.avi")).string()
                                             : output_path;

            cv::VideoWriter writer;
            if (save_results)
            {
                int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                double fps = cap.get(cv::CAP_PROP_FPS);
                if (fps <= 0) fps = 25.0;
                writer.open(out_video_path, fourcc, fps,
                            cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
            }

            cv::Mat frame;
            if (cap.read(frame))
            {
                stats.warmup_iterations = warmup_iterations;
                stats.warmup_total_time_ms = detector.warmup(frame, warmup_iterations);
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            }

            int frame_idx = 0;
            while (cap.read(frame))
            {
                if (frame.empty()) break;
                TimingRecord rec;
                rec.name = "frame_" + std::to_string(frame_idx++);
                auto detections = detector.infer(frame, rec);

                if (save_results && writer.isOpened())
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    detector.draw(frame, detections);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    writer.write(frame);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    rec.draw_time = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    rec.save_time = std::chrono::duration<float, std::milli>(t2 - t1).count();
                }
                rec.total_time = rec.preprocess_time + rec.infer_time + rec.postprocess_time + rec.draw_time + rec.save_time;
                stats.add_record(rec);

                if (frame_idx % 30 == 0)
                {
                    std::cout << "\r[Progress] 帧: " << frame_idx
                              << " | 检测数: " << stats.total_detections
                              << " | 推理: " << std::fixed << std::setprecision(1) << rec.infer_time << "ms" << std::flush;
                }
            }
            std::cout << "\n";
            print_summary(stats);

            std::string report_dir = fs::path(out_video_path).parent_path().string();
            if (report_dir.empty()) report_dir = ".";
            save_reports(report_dir, stats, model_path, input_path);
        }
        else
        {
            if (output_path.empty())
                output_path = input_path + "/results";

            auto image_files = get_image_files(input_path);
            if (image_files.empty())
            {
                std::cerr << "[Error] 未在 " << input_path << " 找到支持的图片文件\n";
                return -1;
            }

            if (save_results)
                fs::create_directories(output_path);

            cv::Mat first = cv::imread(image_files[0].string());
            if (!first.empty())
            {
                stats.warmup_iterations = warmup_iterations;
                stats.warmup_total_time_ms = detector.warmup(first, warmup_iterations);
            }

            int processed = 0;
            for (const auto &img_path : image_files)
            {
                cv::Mat image = cv::imread(img_path.string());
                if (image.empty()) continue;

                TimingRecord rec;
                rec.name = img_path.filename().string();
                auto detections = detector.infer(image, rec);

                if (save_results)
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    detector.draw(image, detections);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    cv::imwrite(output_path + "/" + img_path.filename().string(), image);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    rec.draw_time = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    rec.save_time = std::chrono::duration<float, std::milli>(t2 - t1).count();
                }
                rec.total_time = rec.preprocess_time + rec.infer_time + rec.postprocess_time + rec.draw_time + rec.save_time;
                stats.add_record(rec);

                processed++;
                if (processed % 10 == 0 || processed == (int)image_files.size())
                {
                    std::cout << "\r[Progress] " << processed << "/" << image_files.size()
                              << " | 检测数: " << stats.total_detections
                              << " | 推理: " << std::fixed << std::setprecision(1) << rec.infer_time << "ms" << std::flush;
                }
            }
            std::cout << "\n";
            print_summary(stats);
            save_reports(output_path, stats, model_path, input_path);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] " << e.what() << "\n";
        return -1;
    }

    return 0;
}
