import json
from transformers import AutoTokenizer
from datasets import load_dataset

# Load Flipkart dataset
dataset = load_dataset("KayEe/flipkart_sentiment_analysis")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def preprocess_data(split, tokenized_file, labels_file):
    """
    Preprocess the dataset split by tokenizing the text data and extracting labels.
    
    Args:
        split (str): The dataset split to preprocess (e.g., 'train' or 'test').
        tokenized_file (str): Path to save the tokenized data as a JSON file.
        labels_file (str): Path to save the labels as a JSON file.
    """
    tokenized_data = []
    labels = []
    for example in dataset[split]:
        # Tokenize the text
        tokens = tokenizer(
            example["input"],  # Replace "input" with the correct key if needed
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Save tokenized data
        tokenized_data.append({
            "input_ids": tokens["input_ids"].squeeze().tolist(),
            "attention_mask": tokens["attention_mask"].squeeze().tolist()
        })
        #print(tokenized_data[0])
        # Save label
        #labels.append([example["output"]])  # Replace "output" with the correct key if needed
        print(labels[0])

    # Save tokenized data to JSON
    with open(tokenized_file, "w") as f:
        json.dump(tokenized_data, f, indent=2)

    # Save labels to JSON
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Processed {split} data: {len(tokenized_data)} samples saved.")

if __name__ == "__main__":
    # Define file paths for tokenized data and labels
    tokenized_file = "/mnt/combined/home/parveen/varsha/sentiment_infer/tokenized_flipkart_train.json"
    labels_file = "/mnt/combined/home/parveen/varsha/sentiment_infer/labels_flipkart_train.json"

    # Preprocess and save the train split
    preprocess_data("train", tokenized_file, labels_file)

    print(f"Tokenized data saved to {tokenized_file}")
    print(f"Labels saved to {labels_file}")












#3rd

# from transformers import AutoTokenizer
# from datasets import load_dataset
# import json

# # Load the CardiffNLP tokenizer
# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# def tokenize_dataset(dataset, text_key, output_file):
#     tokenized_data = []
    
#     # Iterate over the dataset and tokenize each text
#     for example in dataset:
#         tokens = tokenizer(example[text_key], padding="max_length", truncation=True, return_tensors="pt")
        
#         # Append tokenized data
#         tokenized_data.append({
#             "input_ids": tokens["input_ids"].tolist()[0],
#             "attention_mask": tokens["attention_mask"].tolist()[0],
#             # Optionally include labels or other fields if needed
#         })

#     # Save all tokenized data to a JSON file
#     with open(output_file, "w") as f:
#         json.dump(tokenized_data, f)

# if __name__ == "__main__":
#     # Load Flipkart sentiment analysis dataset
#     dataset = load_dataset("KayEe/flipkart_sentiment_analysis")
    
#     # Choose the split (train or test)
#     split = "train"
#     if split in dataset:
#         data_split = dataset[split]
#     else:
#         raise ValueError(f"Split '{split}' not found in the dataset.")
    
#     # Define the text key and output file
#     text_key = "input"  # Replace this if another key contains the text
#     # output_file = f"tokenized_flipkart_{split}.json"
#     output_file = "/mnt/combined/home/parveen/varsha/sentiment_infer/tokenized_flipkart_train.json"

    
#     # Tokenize the dataset and save
#     tokenize_dataset(data_split, text_key, output_file)
#     print(f"Tokenized {split} data saved to {output_file}")





# from transformers import AutoTokenizer
# import json

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def tokenize_text(input_text, output_file):
#     # Tokenize the input text
#     tokens = tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    
#     # Prepare data for saving
#     data = {
#         "input_ids": tokens["input_ids"].tolist()[0],
#         "attention_mask": tokens["attention_mask"].tolist()[0],
#     }

#     # Save tokenized data to a JSON file
#     with open(output_file, "w") as f:
#         json.dump(data, f)

# if __name__ == "__main__":
#     input_text = input("Enter text for sentiment analysis: ")
#     output_file = "tokenized_data.json"
#     tokenize_text(input_text, output_file)
#     print(tokenizer.decode(tokenizer(input_text)["input_ids"]))

#     print(f"Tokenized data saved to {output_file}")


###########################################


# from transformers import AutoTokenizer
# from datasets import load_dataset
# import json

# # Load the tokenizer
# #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# def tokenize_dataset(dataset, output_file):
#     tokenized_data = []
    
#     # Iterate over the dataset and tokenize each text
#     for example in dataset:
#         tokens = tokenizer(example['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        
#         # Append tokenized data
#         tokenized_data.append({
#             "input_ids": tokens["input_ids"].tolist()[0],
#             "attention_mask": tokens["attention_mask"].tolist()[0],
#             "label": example["label"],  # Include the label for supervised tasks
#         })

#     # Save all tokenized data to a JSON file
#     with open(output_file, "w") as f:
#         json.dump(tokenized_data, f)

# if __name__ == "__main__":
#     # Load Flipkart sentiment analysis dataset
#     dataset = load_dataset("KayEe/flipkart_sentiment_analysis")
#     print(dataset)
#     # Choose the split (train, test, or validation)
#     split = "train"
#     if split in dataset:
#         data_split = dataset[split]
#     else:
#         raise ValueError(f"Split '{split}' not found in the dataset.")
    
#     output_file = f"tokenized_flipkart_{split}.json"
    
#     # Tokenize the dataset and save
#     tokenize_dataset(data_split, output_file)
#     print(f"Tokenized {split} data saved to {output_file}")
