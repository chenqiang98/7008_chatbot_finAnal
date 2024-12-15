# Loan Enquiry and Credit Risk Analysis Chatbot

This project is a chatbot application designed to assist users with loan enquiries and credit risk analysis. The chatbot interacts with users to gather necessary information, analyze the data, and provide insights on loan eligibility and credit risk.

## Introduction

The Loan Enquiry and Credit Risk Analysis Chatbot is a PyQt5-based application designed to assist users with loan enquiries and credit risk analysis. The chatbot interacts with users through a series of questions to gather necessary information, analyze the data using pre-trained models, and provide insights on loan eligibility and credit risk. The application leverages BERT for sentiment analysis and a custom machine learning model for loan status prediction.

## Usage Instructions

1. **Install Dependencies**: Ensure you have Python and the required libraries installed. You can install the dependencies using the following command:
    ```sh
    pip install -r requirements.txt
    ```

2. **Download the model**: Download our 3 models in the model_link.txt, and place them in correct path:
   ./bert-en/pytorch_model.bin
   ./models/Llama-3.2-1B-Instruct/model.safetensors
   ./selected_epoch.pth

3. **Run the Application**: Execute the `chatbot.py` file to start the chatbot application.
    ```sh
    python chatbot.py
    ```

4. **Interact with the Chatbot**: The chatbot will guide you through a series of questions to gather information about your loan application and credit status. Respond to the chatbot's prompts to receive insights and predictions.

5. **Analyze Results**: The chatbot will analyze your inputs and provide a final report on your loan eligibility and credit risk.

## Example Interaction

```
Bot: Hello! How are you today?
Bot: I'm your consultant on loan enquiry and credit risk analysis.
Bot: Let's start by discussing your reasons for loan application.
You: I need a loan to buy a new car.
Bot: Your reason is positive.
Bot: Let's move on to other information.
Bot: Please input your grade (From A to G).
You: A
Bot: Please input your employer's name.
You: XYZ Corp
...
Bot: I'm analyzing your loan status.
Bot: Please wait for a moment.
Bot: Your credit risk is low.
Bot: I'm ready to give you a final report on loan and credit risk.
Bot: Please type 'report' to get the report.
You: report
Bot: Here's your final report on loan and credit risk.
```

## File Structure

```
.
├── __init__.py
├── .gitignore
├── bert-en/
│   ├── config.json
│   ├── vocab.txt
│   ├── **pytorch_model.bin(download by model_link.txt)**
├── chatbot.py
├── classes_chatbot_finAnal.png
├── dataProcessor.py
├── dataset.py
├── diagrams/
├── llama.py
├── model_link.txt
├── model.py
├── models/
│   ├── DecisionTree.pkl
│   ├── LabelEncoder.pkl
│   ├── Llama-3.2-1B-Instruct/
│   │   ├── .cache/
│   │   ├── .gitattributes
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── **model.safetensors(download by model_link.txt)**
│   │   ├── README.md
│   ├── Llama-3.2-3B-Instruct/
├── packages_chatbot_finAnal.png
├── predict_loan_status.py
├── **selected_epoch.pth(download by model_link.txt)**
├── sentiment_analysis.py
├── README.md
```

## File Descriptions

- `__init__.py`: Initialization file for the package.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `bert-en/`: Directory containing BERT model configuration and vocabulary files.
  - `config.json`: Configuration file for the BERT model.
  - `vocab.txt`: Vocabulary file for the BERT model.
- `chatbot.py`: Main file for the chatbot application. It defines the `ChatBot` class and handles the user interface and conversation logic.
- `classes_chatbot_finAnal.png`: Graphviz DOT file for visualizing chatbot classes.
- `dataProcessor.py`: Contains data processing utilities for preparing data for analysis.
- `dataset.py`: Defines the dataset structure and loading mechanisms.
- `diagrams/`: Directory for storing diagrams related to the project.
- `llama.py`: Contains functions for interacting with the Llama model.
- `model_link.txt`: Contains a link to download the model.
- `model.py`: Defines the machine learning model used for predictions.
- `models/`: Directory containing pre-trained models and related files.
  - `DecisionTree.pkl`: Pre-trained Decision Tree model.
  - `LabelEncoder.pkl`: Pre-trained Label Encoder.
  - `Llama-3.2-1B-Instruct/`: Directory for the Llama 3.2-1B-Instruct model.
    - `.cache/`: Cache directory for the model.
    - `.gitattributes`: Git attributes file for the model.
    - `config.json`: Configuration file for the model.
    - `generation_config.json`: Generation configuration file for the model.
    - `model.safetensors`: Model weights file.
    - `README.md`: Readme file for the model.
  - `Llama-3.2-3B-Instruct/`: Directory for the Llama 3.2-3B-Instruct model.
- `packages_chatbot_finAnal.png`: Graphviz DOT file for visualizing chatbot packages.
- `predict_loan_status.py`: Contains functions for predicting loan status based on user input.
- `selected_epoch.pth`: Pre-trained model weights file.
- `sentiment_analysis.py`: Contains functions for performing sentiment analysis on user input.
