// #include <torch/script.h>
// #include <torch/torch.h>
// #include <iostream>
// #include <fstream>
// #include <nlohmann/json.hpp>

// using Json = nlohmann::json;
//  // Ensure you have a JSON library for C++

// // Function to tokenize input text (read tokenized data from JSON file)
// std::pair<torch::Tensor, torch::Tensor> tokenize_text(const std::string& json_file) {
//     std::ifstream file(json_file);
//     Json::Value root;
//     file >> root;

//     // Convert JSON arrays to tensors
//     std::vector<int64_t> input_ids(root["input_ids"].begin(), root["input_ids"].end());
//     std::vector<int64_t> attention_mask(root["attention_mask"].begin(), root["attention_mask"].end());

//     auto options = torch::TensorOptions().dtype(torch::kInt64);
//     return {torch::tensor(input_ids, options), torch::tensor(attention_mask, options)};
// }

// // Function to perform sentiment analysis
// std::vector<float> analyze_sentiment(torch::jit::script::Module& model, const torch::Tensor& input_ids, 
//                                      const torch::Tensor& attention_mask) {
//     // Perform inference
//     auto outputs = model.forward({input_ids.unsqueeze(0), attention_mask.unsqueeze(0)}).toTensor();

//     // Apply softmax to get probabilities
//     auto probabilities = torch::softmax(outputs, 1);
//     return probabilities.squeeze(0).to(std::vector<float>());
// }

// int main() {
//     try {
//         // Load the TorchScript model
//         torch::jit::script::Module model = torch::jit::load("/mnt/combined/home/parveen/varsha/sentiment_model1.pt");
//         model.eval(); // Switch to evaluation mode

//         // Read tokenized data from JSON file
//         auto tokens = tokenize_text("tokenized_data.json");
//         auto input_ids = tokens.first;
//         auto attention_mask = tokens.second;

//         // Perform sentiment analysis
//         auto probabilities = analyze_sentiment(model, input_ids, attention_mask);

//         // Print the sentiment scores
//         std::cout << "Sentiment Probabilities: [Negative: " << probabilities[0]
//                   << ", Neutral: " << probabilities[1]
//                   << ", Positive: " << probabilities[2] << "]" << std::endl;
//     } catch (const c10::Error& e) {
//         std::cerr << "Error loading the model or running inference: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }

// #include <iostream>
// #include <fstream>
// #include <nlohmann/json.hpp>
// #include <torch/script.h>

// using json = nlohmann::json;

// std::pair<torch::Tensor, torch::Tensor> tokenize_text(const std::string& json_file) {
//     std::cout << "Reading file: " << json_file << std::endl;

//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Failed to open JSON file: " + json_file);
//     }

//     json j;
//     file >> j;

//     // Extract input_ids and attention_mask from JSON
//     auto input_ids = j["input_ids"].get<std::vector<int64_t>>();
//     auto attention_mask = j["attention_mask"].get<std::vector<int64_t>>();

//     auto options = torch::TensorOptions().dtype(torch::kInt64);
//     return {torch::tensor(input_ids, options), torch::tensor(attention_mask, options)};
// } 

// int main() {
//     try {
//         // Load the TorchScript model
//         torch::jit::script::Module model = torch::jit::load("/mnt/combined/home/parveen/varsha/sentiment_model.pt");
//         model.eval(); // Switch to evaluation mode
        
//         // Read tokenized data from JSON file
//         auto tokens = tokenize_text("/mnt/combined/home/parveen/varsha/sentiment_infer/tokenized_data.json");
//         auto input_ids = tokens.first;
//         auto attention_mask = tokens.second;

//         // Perform sentiment analysis
//         auto outputs = model.forward({input_ids.unsqueeze(0), attention_mask.unsqueeze(0)}).toTensor();
//         auto probabilities = torch::softmax(outputs, 1);

//         // Print the sentiment probabilities
//         std::cout << "Sentiment Probabilities: [Negative: " << probabilities[0][0]
//                   << ", Neutral: " << probabilities[0][1]
//                   << ", Positive: " << probabilities[0][2] << "]" << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }




/////



//2nd one
// #include <iostream>
// #include <fstream>
// #include <nlohmann/json.hpp>
// #include <torch/script.h>

// using json = nlohmann::json;

// // Function to tokenize text by reading from a JSON file
// std::pair<torch::Tensor, torch::Tensor> tokenize_text(const std::string& json_file) {
//     std::cout << "Reading file: " << json_file << std::endl;

//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Error: Failed to open JSON file: " + json_file);
//     }

//     json j;
//     file >> j;

//     // Extract input_ids and attention_mask from JSON
//     auto input_ids = j["input_ids"].get<std::vector<int64_t>>();
//     auto attention_mask = j["attention_mask"].get<std::vector<int64_t>>();

//     // Print tokenized data for debugging
//     std::cout << "Input IDs: ";
//     for (const auto& id : input_ids) std::cout << id << " ";
//     std::cout << "\nAttention Mask: ";
//     for (const auto& mask : attention_mask) std::cout << mask << " ";
//     std::cout << std::endl;

//     // Create tensors
//     auto options = torch::TensorOptions().dtype(torch::kInt64);
//     auto input_ids_tensor = torch::tensor(input_ids, options).unsqueeze(0); // Add batch dimension
//     auto attention_mask_tensor = torch::tensor(attention_mask, options).unsqueeze(0); // Add batch dimension

//     return {input_ids_tensor, attention_mask_tensor};
// }

// int main() {
//     try {
//         // Load the TorchScript model
//         const std::string model_path = "/mnt/combined/home/parveen/varsha/sentiment_finetuned_model.pt";
//         torch::jit::script::Module model = torch::jit::load(model_path);
//         model.eval(); // Switch to evaluation mode
//         std::cout << "Model loaded successfully from: " << model_path << std::endl;

//         // Read tokenized data from JSON file
//         const std::string json_path = "/mnt/combined/home/parveen/varsha/sentiment_infer/tokenized_flipkart_train.json";
//         auto tokens = tokenize_text(json_path);
//         auto input_ids = tokens.first;
//         auto attention_mask = tokens.second;

//         // Perform sentiment analysis
//         auto outputs = model.forward({input_ids, attention_mask}).toTensor();

//         // Debugging: Print raw logits
//         std::cout << "Raw Logits: " << outputs << std::endl;

//         // Apply softmax to get probabilities
//         auto probabilities = torch::softmax(outputs, 1);

//         // Print the sentiment probabilities
//         std::cout << "Sentiment Probabilities: [Negative: " << probabilities[0][0].item<float>()
//                   << ", Neutral: " << probabilities[0][1].item<float>()
//                   << ", Positive: " << probabilities[0][2].item<float>() << "]" << std::endl;

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }



//3rd one

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <torch/script.h>
using namespace std;
using json = nlohmann::json;

// Function to load tokenized data from a JSON file
std::vector<std::pair<torch::Tensor, torch::Tensor>> load_tokenized_data(const std::string& json_file) {
    std::cout << "Reading tokenized data from: " << json_file << std::endl;

    std::ifstream file(json_file);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Failed to open JSON file: " + json_file);
    }

    json dataset;
    file >> dataset;

    std::vector<std::pair<torch::Tensor, torch::Tensor>> tokenized_data;

    for (const auto& example : dataset) {
        // Extract input_ids and attention_mask
        auto input_ids = example["input_ids"].get<std::vector<int64_t>>();
        auto attention_mask = example["attention_mask"].get<std::vector<int64_t>>();

        // Create tensors
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        auto input_ids_tensor = torch::tensor(input_ids, options).unsqueeze(0); // Add batch dimension
        auto attention_mask_tensor = torch::tensor(attention_mask, options).unsqueeze(0); // Add batch dimension

        tokenized_data.emplace_back(input_ids_tensor, attention_mask_tensor);
    }

    return tokenized_data;
}

// Function to load labels from a JSON file
// std::vector<string> load_labels(const std::string& labels_file) {
//     std::cout << "Reading labels from: " << labels_file << std::endl;

//     std::ifstream file(labels_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Error: Failed to open labels JSON file: " + labels_file);
//     }

//     json labels_json;
//     file >> labels_json;

//     return labels_json.get<std::vector<string>>();
// }
std::vector<int> load_labels(const std::string& labels_file) {
    std::cout << "Reading labels from: " << labels_file << std::endl;

    std::ifstream file(labels_file);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Failed to open labels JSON file: " + labels_file);
    }

    json labels_json;
    file >> labels_json;

    // Define the mapping from string labels to numerical values
    std::map<std::string, int> label_mapping = {
        {"negative", 0},
        {"neutral", 1},
        {"positive", 2}
    };

    // Convert string labels to integers using the mapping
    std::vector<int> labels;
    for (const auto& label : labels_json) {
        auto it = label_mapping.find(label.get<std::string>());
        if (it != label_mapping.end()) {
            labels.push_back(it->second);
        } else {
            throw std::runtime_error("Error: Unknown label encountered: " + label.get<std::string>());
        }
    }

    return labels;
}

int main() {
    try {
        // Load the TorchScript model
        const std::string model_path = "/mnt/combined/home/parveen/varsha/sentiment_finetuned_model.pt";
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval(); // Switch to evaluation mode
        std::cout << "Model loaded successfully from: " << model_path << std::endl;

        //Paths to tokenized data and labels
        const std::string tokenized_file = "/mnt/combined/home/parveen/varsha/sentiment_infer/tokenized_flipkart_train.json";
        const std::string labels_file = "/mnt/combined/home/parveen/varsha/sentiment_infer/labels_flipkart_train.json";

        // Load tokenized data and labels
        auto tokenized_data = load_tokenized_data(tokenized_file);
        auto labels = load_labels(labels_file);
        //cout<<"Got the labesl\n";
        // Check if labels match the tokenized data size
        if (labels.size() != tokenized_data.size()) {
            throw std::runtime_error("Mismatch between labels and tokenized data size.");
        }

        // Perform sentiment analysis and calculate accuracy
        int correct_predictions = 0;
        for (size_t i = 0; i < tokenized_data.size(); ++i) {
            //cout<<"IN the for loop\n";
            auto input_ids = tokenized_data[i].first;
            auto attention_mask = tokenized_data[i].second;

            // Get model output
            auto outputs = model.forward({input_ids, attention_mask}).toTensor();

            // Get predicted label
            auto predicted_label = outputs.argmax(1).item<int>();
            //cout<<predicted_label<<" pre "<<labels[i]<<"labell\n";
            // Compare with ground truth label
            if (predicted_label == labels[i]) {
                ++correct_predictions;
            }
        }

        // Calculate accuracy
        float accuracy = static_cast<float>(correct_predictions) / labels.size() * 100.0f;
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;



        //


        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }


    //

    
    return 0;
}
