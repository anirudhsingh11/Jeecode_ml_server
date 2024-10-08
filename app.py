from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.models import Sequential
from keras.losses import mean_squared_error
app = Flask(__name__)

# Load the model, but specify the loss manually
lstm_model = load_model('model.h5', compile=False)  # Load the model without compiling first

# Compile the model again with the correct loss function
lstm_model.compile(optimizer='adam', loss=mean_squared_error)

# Load pre-trained models
#lstm_model = load_model('model.h5')  # Update with correct model path
prediction_model = joblib.load('prediction_model.pkl')  # Update if required
df = pd.read_csv('processed_df.csv')  # Load dataset
# Create vocabulary and word-to-index mapping for 'PredictedQuestion'

#embeddingsP = np.load('embeddingsP (1).npy')  # Precomputed embeddings
def process():
    global embeddingsP
    # Create vocabulary and word-to-index mapping for 'PredictedQuestion'
    vocab_pred = set(word for text in df['PredictedQuestion'] if isinstance(text, str) for word in text.split())
    vocab_pred_size = len(vocab_pred) + 1
    word_index_pred = {word: index + 1 for index, word in enumerate(vocab_pred)}

    # Convert text to sequences and pad for 'PredictedQuestion'
    sequences_pred = [[word_index_pred[word] for word in text.split()] for text in df['PredictedQuestion'] if isinstance(text, str)]
    max_length_pred = max(len(seq) for seq in sequences_pred)
    padded_sequences_pred = pad_sequences(sequences_pred, maxlen=max_length_pred, padding='post')

    # Convert padded sequences to NumPy array
    padded_sequences_pred = np.array(padded_sequences_pred)

    # Define embedding and LSTM layers for 'PredictedQuestion'
    embedding_dim = 128
    embedding_layer_pred = Embedding(vocab_pred_size, embedding_dim)
    lstm_layer_pred = LSTM(128)

    # Generate embeddings
    embeddingsP = lstm_layer_pred(embedding_layer_pred(padded_sequences_pred))

    import tensorflow as tf
    # Convert embeddingsP to a NumPy array before reshaping
    embeddingsP = embeddingsP.numpy() if isinstance(embeddingsP, tf.Tensor) else embeddingsP

    # Flatten and reshape embeddingsP to expected dimensions
    n1 = 131
    total_elements = embeddingsP.size
    expected_rows = total_elements // n1
    if total_elements % n1 > 0:
        embeddingsP = embeddingsP.flatten()[:expected_rows * n1]
        embeddingsP = embeddingsP.reshape(expected_rows, n1)
    else:
        embeddingsP = embeddingsP.reshape(expected_rows, n1)
# Function to preprocess and tokenize input question
def preprocess_question(input_question):
    tokenized_input_question = input_question.split()
    vocab_ques = set(tokenized_input_question)
    vocab_ques_size = len(vocab_ques) + 1
    word_to_index_ques = {word: index + 1 for index, word in enumerate(vocab_ques)}
    indexed_input_question = [word_to_index_ques[word] for word in tokenized_input_question]
    max_length_ques = 10
    padded_input_question = pad_sequences([indexed_input_question], maxlen=max_length_ques, padding='post')
    return padded_input_question, vocab_ques_size

# Function to normalize additional features
def normalize_additional_features(points, level_student, difficulty):
    normalized_points = (points - 1000) / (10000 - 1000)
    normalized_level_student = level_student / 10.0
    difficulty_mapping = {'easy': 0, 'medium': 0.5, 'hard': 1}
    normalized_difficulty = difficulty_mapping[difficulty]
    return np.array([[normalized_points, normalized_level_student, normalized_difficulty]])
@app.route('/')
def home():
    return "Welcome to Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    process()
    data = request.get_json()
    
    # Extract features from the input
    input_question = data['question']
    points = data['points']
    level_student = data['level_student']
    difficulty = data['difficulty']
        

    # Select a random question from the dataset
    # random_index = np.random.randint(0, len(df))
    # input_question = df['Question'].iloc[random_index]
    # print("Input Question:", input_question)
    #input_question = "$\int\left(1+x+\frac{x^{2}}{2!}+\frac{x^{3}}{3!}+\ldots \ldots ..\right) d x=$"

    # Tokenize the selected question
    tokenized_input_question = input_question.split()

    # Create vocabulary and word-to-index mapping for tokenized question
    vocab_ques = set(tokenized_input_question)
    vocab_ques_size = len(vocab_ques) + 1

    # Map words to indices
    word_to_index_ques = {word: index + 1 for index, word in enumerate(vocab_ques)}

    # Convert question to numerical indices
    indexed_input_question = [word_to_index_ques[word] for word in tokenized_input_question]

    # Pad the sequence to the max length used in training
    max_length_ques = 10
    padded_input_question = pad_sequences([indexed_input_question], maxlen=max_length_ques, padding='post')
    # Define model to generate embeddings for input question
    embedding_dim = 128
    model = Sequential([
        Embedding(vocab_ques_size, embedding_dim, input_length=max_length_ques),
        LSTM(128)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Generate embeddings for the input question
    input_question_embedding = model.predict(padded_input_question)

    # Normalize Points (range 1000 to 10000)
    normalized_points = (points - 1000) / (10000 - 1000)

    # Normalize Level_Student (range 1 to 10)
    normalized_level_student = level_student / 10.0

    # Map difficulty levels to numeric values
    difficulty_mapping = {'easy': 0, 'medium': 0.5, 'hard': 1}
    normalized_difficulty = difficulty_mapping[difficulty]

    # Combine the normalized features into a single array
    additional_features = np.array([[normalized_points, normalized_level_student, normalized_difficulty]])

    # Concatenate question embeddings with normalized features to create combined input
    combined_input = np.concatenate([input_question_embedding, additional_features], axis=1)
    print("Combined Input Shape:", combined_input.shape)

    
    # Predict embeddings for input question
    predicted_embeddings_for_input = prediction_model.predict(combined_input)
    
    # Calculate cosine similarity between predicted and existing embeddings
    similarity_scores = cosine_similarity(predicted_embeddings_for_input, embeddingsP)
    
    # Find the most similar question
    most_similar_index = np.argmax(similarity_scores)
    predicted_question_id = df['Question_ID'].iloc[most_similar_index]
    predicted_question = df[df['Question_ID'] == predicted_question_id]['Question'].iloc[0]
    predicted_question_rating = df[df['Question_ID'] == predicted_question_id]['Points'].iloc[0]
    
    # Return predicted question as JSON response
    return jsonify({
        'predicted_question_id': predicted_question_id,
        'predicted_question': predicted_question,
        'rating': predicted_question_rating
    })

if __name__ == '__main__':
    app.run(debug=True)


