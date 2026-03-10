#include <chrono>
#include <cctype>
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
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tstring.h"

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
    fs::path saved_model_path;
    fs::path labels_path;
    std::string input_tensor_name;
    std::string class_ids_tensor_name;
    std::string scores_tensor_name;
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

std::string StripTensorPort(const std::string& tensor_name) {
    const auto colon = tensor_name.rfind(':');
    if (colon == std::string::npos) {
        return tensor_name;
    }
    return tensor_name.substr(0, colon);
}

int TensorPort(const std::string& tensor_name) {
    const auto colon = tensor_name.rfind(':');
    if (colon == std::string::npos) {
        return 0;
    }
    return std::stoi(tensor_name.substr(colon + 1));
}

class TensorFlowBackend final : public InferenceBackend {
public:
    explicit TensorFlowBackend(const fs::path& repo_root)
        : repo_root_(repo_root),
          status_(TF_NewStatus()),
          graph_(TF_NewGraph()),
          session_options_(TF_NewSessionOptions()),
          meta_graph_(TF_NewBuffer()) {
        try {
            metadata_ = LoadMetadata();
            labels_ = LoadLabels(metadata_.labels_path);
            LoadSavedModel();
            ready_ = true;
        } catch (const std::exception& error) {
            ready_ = false;
            startup_error_ = error.what();
        }
    }

    ~TensorFlowBackend() override {
        if (session_ != nullptr) {
            TF_CloseSession(session_, status_);
            TF_DeleteSession(session_, status_);
        }
        if (meta_graph_ != nullptr) {
            TF_DeleteBuffer(meta_graph_);
        }
        if (session_options_ != nullptr) {
            TF_DeleteSessionOptions(session_options_);
        }
        if (graph_ != nullptr) {
            TF_DeleteGraph(graph_);
        }
        if (status_ != nullptr) {
            TF_DeleteStatus(status_);
        }
    }

    InferenceResult Predict(const fs::path& image_path) override {
        if (!ready_) {
            return {
                false,
                "",
                0.0F,
                startup_error_.empty() ? "TensorFlow backend is not ready." : startup_error_,
            };
        }

        try {
            const std::vector<char> image_bytes = ReadBinaryFile(image_path);
            std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> input_tensor(
                CreateStringTensor(image_bytes),
                TF_DeleteTensor
            );

            TF_Tensor* output_tensors[2] = {nullptr, nullptr};
            TF_Tensor* input_tensors[1] = {input_tensor.get()};
            TF_Output inputs[1] = {input_};
            TF_Output outputs[2] = {class_ids_output_, scores_output_};

            {
                std::lock_guard<std::mutex> lock(session_mutex_);
                TF_SessionRun(
                    session_,
                    nullptr,
                    inputs,
                    input_tensors,
                    1,
                    outputs,
                    output_tensors,
                    2,
                    nullptr,
                    0,
                    nullptr,
                    status_
                );
                CheckStatus("TF_SessionRun");
            }

            std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> class_ids_tensor(
                output_tensors[0],
                TF_DeleteTensor
            );
            std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> scores_tensor(
                output_tensors[1],
                TF_DeleteTensor
            );

            const auto* class_ids =
                static_cast<const int32_t*>(TF_TensorData(class_ids_tensor.get()));
            const auto* scores =
                static_cast<const float*>(TF_TensorData(scores_tensor.get()));

            const int class_id = class_ids[0];
            if (class_id < 0 || static_cast<size_t>(class_id) >= labels_.size()) {
                throw std::runtime_error("Predicted class id is out of label range.");
            }

            const float confidence = scores[class_id];
            return {true, labels_[class_id], confidence, ""};
        } catch (const std::exception& error) {
            return {false, "", 0.0F, error.what()};
        }
    }

private:
    ServingMetadata LoadMetadata() const {
        const fs::path manifest_path = repo_root_ / "main" / "models" / "model_manifest.json";
        if (!fs::exists(manifest_path)) {
            throw std::runtime_error(
                "Model manifest not found. Run `python -m main.export_serving --model-name ButterflyC` first."
            );
        }

        const auto manifest_json = crow::json::load(ReadTextFile(manifest_path));
        if (!manifest_json) {
            throw std::runtime_error("Failed to parse model manifest JSON.");
        }

        const auto serving = manifest_json["serving"];
        const auto output_tensor_names = serving["output_tensor_names"];

        return {
            ResolveRepoPath(repo_root_, std::string(manifest_json["saved_model_path"].s())),
            ResolveRepoPath(repo_root_, std::string(manifest_json["labels_path"].s())),
            std::string(serving["input_tensor_name"].s()),
            std::string(output_tensor_names["class_ids"].s()),
            std::string(output_tensor_names["scores"].s()),
        };
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

    void LoadSavedModel() {
        const char* tags[] = {"serve"};
        session_ = TF_LoadSessionFromSavedModel(
            session_options_,
            nullptr,
            metadata_.saved_model_path.string().c_str(),
            tags,
            1,
            graph_,
            meta_graph_,
            status_
        );
        CheckStatus("TF_LoadSessionFromSavedModel");
        if (session_ == nullptr) {
            throw std::runtime_error("TensorFlow session was not created.");
        }

        input_ = ResolveOutput(metadata_.input_tensor_name);
        class_ids_output_ = ResolveOutput(metadata_.class_ids_tensor_name);
        scores_output_ = ResolveOutput(metadata_.scores_tensor_name);
    }

    TF_Output ResolveOutput(const std::string& tensor_name) {
        TF_Operation* operation =
            TF_GraphOperationByName(graph_, StripTensorPort(tensor_name).c_str());
        if (operation == nullptr) {
            throw std::runtime_error("TensorFlow graph operation not found: " + tensor_name);
        }

        return TF_Output{operation, TensorPort(tensor_name)};
    }

    TF_Tensor* CreateStringTensor(const std::vector<char>& value) const {
        const int64_t dims[] = {1};
        TF_Tensor* tensor =
            TF_AllocateTensor(TF_STRING, dims, 1, sizeof(TF_TString));
        if (tensor == nullptr) {
            throw std::runtime_error("Failed to allocate TF_STRING tensor.");
        }

        auto* tensor_data = static_cast<TF_TString*>(TF_TensorData(tensor));
        TF_StringInit(&tensor_data[0]);
        TF_StringCopy(&tensor_data[0], value.data(), value.size());
        return tensor;
    }

    void CheckStatus(const std::string& action) const {
        if (TF_GetCode(status_) != TF_OK) {
            throw std::runtime_error(action + ": " + TF_Message(status_));
        }
    }

    fs::path repo_root_;
    bool ready_ = false;
    std::string startup_error_;
    ServingMetadata metadata_;
    std::vector<std::string> labels_;
    TF_Status* status_ = nullptr;
    TF_Graph* graph_ = nullptr;
    TF_SessionOptions* session_options_ = nullptr;
    TF_Buffer* meta_graph_ = nullptr;
    TF_Session* session_ = nullptr;
    TF_Output input_{};
    TF_Output class_ids_output_{};
    TF_Output scores_output_{};
    std::mutex session_mutex_;
};

}  // namespace

int main() {
    const fs::path repo_root = fs::current_path();
    const fs::path templates_dir = repo_root / "app" / "templates";
    const fs::path upload_dir = repo_root / "app" / "upload";
    fs::create_directories(upload_dir);

    TensorFlowBackend backend(repo_root);
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
