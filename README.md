# Word2Vec Model for Product Review Classification

This project involves building a Word2Vec (W2V) model for classifying product reviews. The model meets the following criteria:

### Requirements:
1. The model must operate on pre-processed review data.
2. The model should be compared to a classic Bag of Words (BoW) model using TF-IDF transformation in terms of classification performance.
3. For classification, at least 15 training reviews and 45 validation reviews must be used. The reviews should be equally distributed between positive, negative, and neutral categories.
4. The classifier used for training is a Multilayer Perceptron (MLP).

### Prerequisites:
- Python must be installed on your system.

### Steps to Run:
1. Create a virtual environment using `python -m venv venv`.
2. Install the required dependencies by running:  
   ```  
   pip install -r requirements.txt  
   ```
3. Execute the model by running:  
   ```  
   python main.py  
   ```

### Notes:
- Ensure the dataset has been pre-processed before running the model.
- The project includes a comparison between the Word2Vec model and a traditional Bag of Words model to evaluate the classification performance.