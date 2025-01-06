import pandas as pd
import numpy as np
import regex as re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, RepeatVector, TimeDistributed, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load the datasets
    english = pd.read_csv('/Users/amnaalmurshda/Desktop/FLASK_Trans/en.csv', header=None, names=['English'])
    french = pd.read_csv('/Users/amnaalmurshda/Desktop/FLASK_Trans/fr.csv', header=None, names=['French'])

    # Combine the datasets
    df = pd.concat([english, french], axis=1)
    df.columns = ['English', 'French']

    # Function to clean text
    def clean_text(text):
        text = re.sub(r'[.!?:;,]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    # Apply the function to the datasets
    df['English'] = df['English'].apply(clean_text)
    df['French'] = df['French'].apply(clean_text)

    return df

def tokenize_and_pad(df):
    eng_tokenizer = Tokenizer(oov_token='<OOV>')
    eng_tokenizer.fit_on_texts(df['English'])
    eng_sequences = eng_tokenizer.texts_to_sequences(df['English'])
    max_length_eng = max([len(seq) for seq in eng_sequences])
    eng_padded = pad_sequences(eng_sequences, maxlen=max_length_eng, padding='post')

    fr_tokenizer = Tokenizer(oov_token='<OOV>')
    fr_tokenizer.fit_on_texts(df['French'])
    fr_sequences = fr_tokenizer.texts_to_sequences(df['French'])
    max_length_fr = max([len(seq) for seq in fr_sequences])
    fr_padded = pad_sequences(fr_sequences, maxlen=max_length_fr, padding='post')

    return eng_padded, fr_padded, eng_tokenizer, fr_tokenizer, max_length_eng, max_length_fr

# Load and preprocess the data
df = load_and_preprocess_data()

# Tokenize and pad the sequences
eng_padded, fr_padded, eng_tokenizer, fr_tokenizer, max_length_eng, max_length_fr = tokenize_and_pad(df)

# Split data into training and validation sets
eng_train, eng_val, fr_train, fr_val = train_test_split(eng_padded, fr_padded, test_size=0.2, random_state=42)

# Build the GRU-based Seq2Seq model
vocab_size_eng = len(eng_tokenizer.word_index)
vocab_size_fr = len(fr_tokenizer.word_index)

model = Sequential()
model.add(Embedding(vocab_size_eng + 1, 128, input_length=max_length_eng))
model.add(Bidirectional(GRU(64, return_sequences=False)))
model.add(RepeatVector(max_length_fr))
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(TimeDistributed(Dense(vocab_size_fr + 1, activation="softmax")))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with EarlyStopping
french_train_output = np.expand_dims(fr_train, -1)
french_val_output = np.expand_dims(fr_val, -1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    eng_train, french_train_output,
    batch_size=64,
    epochs=50,
    validation_data=(eng_val, french_val_output),
    callbacks=[early_stopping]
)

# Save the model
model.save('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/gru_seq2seq_model.h5')

# Save the tokenizers
with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/eng_tokenizer.pkl', 'wb') as file:
    pickle.dump(eng_tokenizer, file)
with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/fr_tokenizer.pkl', 'wb') as file:
    pickle.dump(fr_tokenizer, file)

print("Model and tokenizers saved successfully.")
