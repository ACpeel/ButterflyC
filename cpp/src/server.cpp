#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CROW_STATIC_DIRECTORY "app/templates/static/"
#include <crow.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

struct InferenceResult {
    bool ok = false;
    std::string label;
    float confidence = 0.0F;
    std::string message;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual InferenceResult Predict(const fs::path& image_path) = 0;
};

struct ServingMetadata {
    fs::path onnx_model_path;
    fs::path labels_path;
    int image_size = 224;
    std::string input_name;
    std::vector<std::string> output_names;
    bool output_is_probabilities = false;
};

std::string ReadTextFile(const fs::path& file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::vector<char> ReadBinaryFile(const fs::path& file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open image file: " + file_path.string());
    }

    return std::vector<char>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()
    );
}

fs::path ResolveRepoPath(const fs::path& repo_root, const std::string& raw_path) {
    const fs::path candidate(raw_path);
    if (candidate.is_absolute()) {
        return candidate;
    }
    return (repo_root / candidate).lexically_normal();
}

std::string DetectMimeType(const fs::path& file_path) {
    const std::string extension = file_path.extension().string();
    if (extension == ".html") {
        return "text/html; charset=utf-8";
    }
    if (extension == ".css") {
        return "text/css; charset=utf-8";
    }
    if (extension == ".js") {
        return "application/javascript; charset=utf-8";
    }
    if (extension == ".jpg" || extension == ".jpeg") {
        return "image/jpeg";
    }
    if (extension == ".png") {
        return "image/png";
    }
    if (extension == ".json") {
        return "application/json; charset=utf-8";
    }
    return "application/octet-stream";
}

std::optional<fs::path> ResolveSafePath(
    const fs::path& root,
    const std::string& relative_path
) {
    const fs::path absolute_root = fs::absolute(root).lexically_normal();
    const fs::path candidate = fs::absolute(absolute_root / relative_path).lexically_normal();

    const auto root_string = absolute_root.string();
    const auto candidate_string = candidate.string();
    if (candidate_string.rfind(root_string, 0) != 0) {
        return std::nullopt;
    }

    return candidate;
}

crow::response MakeFileResponse(const fs::path& file_path) {
    if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) {
        return crow::response(404, "file not found");
    }

    std::ifstream file_stream(file_path, std::ios::binary);
    std::ostringstream buffer;
    buffer << file_stream.rdbuf();

    crow::response response(buffer.str());
    response.code = 200;
    response.set_header("Content-Type", DetectMimeType(file_path));
    return response;
}

std::string SanitizeFilename(const std::string& filename) {
    std::string sanitized;
    sanitized.reserve(filename.size());

    for (unsigned char ch : filename) {
        if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
    }

    if (sanitized.empty()) {
        return "upload.bin";
    }

    return sanitized;
}

std::string BuildStoredFilename(const std::string& original_name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string(stamp) + "_" + SanitizeFilename(original_name);
}

std::string UrlEncode(const std::string& input) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(input.size() * 3);

    for (unsigned char ch : input) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back(static_cast<char>(ch));
        } else {
            encoded.push_back('%');
            encoded.push_back(hex[(ch >> 4) & 0x0F]);
            encoded.push_back(hex[ch & 0x0F]);
        }
    }

    return encoded;
}

crow::json::wvalue ErrorPayload(const std::string& message) {
    crow::json::wvalue payload;
    payload["error"] = message;
    return payload;
}

std::vector<float> PreprocessImageBytes(const std::vector<char>& bytes, int image_size) {
    static constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
    static constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};

    cv::Mat buffer(1, static_cast<int>(bytes.size()), CV_8U, const_cast<char*>(bytes.data()));
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to decode image bytes.");
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(image_size, image_size));
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    const size_t plane_size = static_cast<size_t>(image_size) * image_size;
    std::vector<float> chw(3 * plane_size);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - kMean[c]) * (1.0F / kStd[c]);
        if (!channels[c].isContinuous()) {
            channels[c] = channels[c].clone();
        }
        std::memcpy(
            chw.data() + c * plane_size,
            channels[c].data,
            plane_size * sizeof(float)
        );
    }
    return chw;
}

std::vector<float> Softmax(const float* logits, size_t size) {
    if (size == 0) {
        return {};
    }
    const float max_logit = *std::max_element(logits, logits + size);
    std::vector<float> probs(size);
    float sum = 0.0F;
    for (size_t i = 0; i < size; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum <= 0.0F) {
        return probs;
    }
    for (size_t i = 0; i < size; ++i) {
        probs[i] /= sum;
    }
    return probs;
}

class OnnxBackend final : public InferenceBackend {
public:
    explicit OnnxBackend(const fs::path& repo_root)
        : repo_root_(repo_root),
          env_(ORT_LOGGING_LEVEL_WARNING, "butterflyc") {
        try {
            metadata_ = LoadMetadata();
            labels_ = LoadLabels(metadata_.labels_path);
            LoadSession();
            ready_ = true;
        } catch (const std::exception& error) {
            ready_ = false;
            startup_error_ = error.what();
        }
    }

    ~OnnxBackend() override = default;

    InferenceResult Predict(const fs::path& image_path) override {
        if (!ready_) {
            return {
                false,
                "",
                0.0F,
                startup_error_.empty() ? "ONNX backend is not ready." : startup_error_,
            };
        }

        try {
            const std::vector<char> image_bytes = ReadBinaryFile(image_path);
            std::vector<float> input = PreprocessImageBytes(
                image_bytes,
                metadata_.image_size
            );
            const std::array<int64_t, 4> input_shape = {
                1,
                3,
                metadata_.image_size,
                metadata_.image_size,
            };
            Ort::MemoryInfo memory_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input.data(),
                input.size(),
                input_shape.data(),
                input_shape.size()
            );

            const char* input_name = input_name_.c_str();
            std::vector<const char*> output_names;
            output_names.reserve(output_names_.size());
            for (const auto& name : output_names_) {
                output_names.push_back(name.c_str());
            }

            std::lock_guard<std::mutex> lock(session_mutex_);
            auto outputs = session_.Run(
                Ort::RunOptions{nullptr},
                &input_name,
                &input_tensor,
                1,
                output_names.data(),
                output_names.size()
            );
            if (outputs.empty()) {
                throw std::runtime_error("ONNX inference returned no outputs.");
            }

            const float* scores = outputs[0].GetTensorData<float>();
            const size_t num_classes = labels_.size();
            if (num_classes == 0) {
                throw std::runtime_error("Labels list is empty.");
            }

            size_t best_index = 0;
            float best_value = scores[0];
            for (size_t i = 1; i < num_classes; ++i) {
                if (scores[i] > best_value) {
                    best_value = scores[i];
                    best_index = i;
                }
            }

            float confidence = best_value;
            if (!metadata_.output_is_probabilities) {
                const auto probs = Softmax(scores, num_classes);
                confidence = probs[best_index];
            }
            return {true, labels_[best_index], confidence, ""};
        } catch (const std::exception& error) {
            return {false, "", 0.0F, error.what()};
        }
    }

private:
    ServingMetadata LoadMetadata() const {
        const fs::path manifest_path = repo_root_ / "main" / "models" / "model_manifest.json";
        if (!fs::exists(manifest_path)) {
            throw std::runtime_error(
                "Model manifest not found. Run `python -m main.export_onnx --model-name ButterflyC` first."
            );
        }

        const auto manifest_json = crow::json::load(ReadTextFile(manifest_path));
        if (!manifest_json) {
            throw std::runtime_error("Failed to parse model manifest JSON.");
        }
        ServingMetadata metadata;
        if (manifest_json.has("onnx_model_path")) {
            metadata.onnx_model_path =
                ResolveRepoPath(repo_root_, std::string(manifest_json["onnx_model_path"].s()));
        } else {
            std::string model_name = "ButterflyC";
            if (manifest_json.has("default_model")) {
                model_name = std::string(manifest_json["default_model"].s());
            }
            metadata.onnx_model_path =
                repo_root_ / "main" / "models" / (model_name + ".onnx");
        }
        metadata.labels_path =
            ResolveRepoPath(repo_root_, std::string(manifest_json["labels_path"].s()));
        if (manifest_json.has("image_size")) {
            metadata.image_size = manifest_json["image_size"].i();
        }
        if (manifest_json.has("onnx_input_name")) {
            metadata.input_name = std::string(manifest_json["onnx_input_name"].s());
        }
        if (manifest_json.has("onnx_output_names")) {
            const auto outputs = manifest_json["onnx_output_names"];
            metadata.output_names.reserve(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                metadata.output_names.emplace_back(std::string(outputs[i].s()));
            }
        }
        if (manifest_json.has("onnx_output_type")) {
            const std::string output_type =
                std::string(manifest_json["onnx_output_type"].s());
            metadata.output_is_probabilities = (output_type == "probabilities");
        }
        return metadata;
    }

    std::vector<std::string> LoadLabels(const fs::path& labels_path) const {
        const auto labels_json = crow::json::load(ReadTextFile(labels_path));
        if (!labels_json) {
            throw std::runtime_error("Failed to parse labels JSON.");
        }

        std::vector<std::string> labels;
        const auto classes = labels_json["classes"];
        labels.reserve(classes.size());
        for (size_t index = 0; index < classes.size(); ++index) {
            labels.emplace_back(std::string(classes[index].s()));
        }
        if (labels.empty()) {
            throw std::runtime_error("Labels list is empty.");
        }
        return labels;
    }

    void LoadSession() {
        if (!fs::exists(metadata_.onnx_model_path)) {
            throw std::runtime_error(
                "ONNX model not found: " + metadata_.onnx_model_path.string()
            );
        }
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, metadata_.onnx_model_path.c_str(), session_options_);

        if (metadata_.input_name.empty()) {
            auto input_name = session_.GetInputNameAllocated(0, allocator_);
            input_name_ = input_name.get();
        } else {
            input_name_ = metadata_.input_name;
        }
        if (metadata_.output_names.empty()) {
            const size_t output_count = session_.GetOutputCount();
            if (output_count == 0) {
                throw std::runtime_error("ONNX model has no outputs.");
            }
            for (size_t i = 0; i < output_count; ++i) {
                auto output_name = session_.GetOutputNameAllocated(i, allocator_);
                output_names_.emplace_back(output_name.get());
            }
        } else {
            output_names_ = metadata_.output_names;
        }
    }

    fs::path repo_root_;
    bool ready_ = false;
    std::string startup_error_;
    ServingMetadata metadata_;
    std::vector<std::string> labels_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::mutex session_mutex_;
};

}  // namespace

int main() {
    const fs::path repo_root = fs::current_path();
    const fs::path templates_dir = repo_root / "app" / "templates";
    const fs::path upload_dir = repo_root / "app" / "upload";
    fs::create_directories(upload_dir);

    OnnxBackend backend(repo_root);
    crow::SimpleApp app;

    CROW_ROUTE(app, "/healthz")([] {
        return crow::response(200, "ok");
    });

    CROW_ROUTE(app, "/")([templates_dir] {
        return MakeFileResponse(templates_dir / "index.html");
    });

    CROW_ROUTE(app, "/result")([templates_dir] {
        return MakeFileResponse(templates_dir / "result.html");
    });

    CROW_ROUTE(app, "/butterfly")([] {
        crow::response response(302);
        response.set_header(
            "Location",
            "https://www.inaturalist.org/taxa/47224-Papilionoidea"
        );
        return response;
    });

    CROW_ROUTE(app, "/uploaded/<path>")([upload_dir](const std::string& requested_path) {
        const auto safe_path = ResolveSafePath(upload_dir, requested_path);
        if (!safe_path.has_value()) {
            return crow::response(400, "invalid path");
        }
        return MakeFileResponse(*safe_path);
    });

    CROW_ROUTE(app, "/ur")
        .methods(crow::HTTPMethod::Post)
        ([upload_dir, &backend](const crow::request& req) {
            crow::multipart::message multipart(req);
            const auto file_part = multipart.get_part_by_name("file");
            if (file_part.body.empty()) {
                crow::response response(400, ErrorPayload("missing file").dump());
                response.set_header("Content-Type", "application/json; charset=utf-8");
                return response;
            }

            auto disposition = file_part.get_header_object("Content-Disposition");
            auto filename_it = disposition.params.find("filename");
            const std::string original_name =
                filename_it != disposition.params.end() ? filename_it->second : "upload.bin";
            const std::string stored_name = BuildStoredFilename(original_name);
            const fs::path stored_path = upload_dir / stored_name;

            std::ofstream output(stored_path, std::ios::binary);
            output.write(file_part.body.data(), static_cast<std::streamsize>(file_part.body.size()));
            output.close();

            const InferenceResult result = backend.Predict(stored_path);
            if (!result.ok) {
                crow::response response(503, ErrorPayload(result.message).dump());
                response.set_header("Content-Type", "application/json; charset=utf-8");
                return response;
            }

            crow::json::wvalue payload;
            payload["redirect"] =
                "/result?image=" + UrlEncode(stored_name) +
                "&category=" + UrlEncode(result.label);
            payload["image"] = stored_name;
            payload["category"] = result.label;
            payload["confidence"] = result.confidence;

            crow::response response(200, payload.dump());
            response.set_header("Content-Type", "application/json; charset=utf-8");
            return response;
        });

    const char* port_env = std::getenv("PORT");
    const std::uint16_t port = port_env == nullptr
        ? 8091
        : static_cast<std::uint16_t>(std::stoi(port_env));

    app.loglevel(crow::LogLevel::Info);
    app.port(port).multithreaded().run();
    return 0;
}
