import pandas as pd
import numpy as np
import regex as re
import pickle
import uuid
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess_data():
    # Load the datasets
    english = pd.read_csv('/Users/amnaalmurshda/Desktop/FLASK_Trans/en.csv', header=None, names=['English'])
    french = pd.read_csv('/Users/amnaalmurshda/Desktop/FLASK_Trans/fr.csv', header=None, names=['French'])

    # Combine the datasets
    df = pd.concat([english, french], axis=1)
    df.columns = ['English', 'French']

    # Function to remove punctuation
    def remove_punctuation(text):
        return re.sub(r'[.!?:;,]', '', text)

    # Apply the function to the datasets
    df['English'] = df['English'].apply(lambda x: remove_punctuation(x))
    df['French'] = df['French'].apply(lambda x: remove_punctuation(x))

    return df

def tokenize_and_pad(df):
    eng_tokenizer = Tokenizer()
    eng_tokenizer.fit_on_texts(df['English'])
    eng_sequences = eng_tokenizer.texts_to_sequences(df['English'])
    max_length_eng = max([len(seq) for seq in eng_sequences])
    eng_padded = pad_sequences(eng_sequences, maxlen=max_length_eng, padding='post')

    fr_tokenizer = Tokenizer()
    fr_tokenizer.fit_on_texts(df['French'])
    fr_sequences = fr_tokenizer.texts_to_sequences(df['French'])
    max_length_fr = max([len(seq) for seq in fr_sequences])
    fr_padded = pad_sequences(fr_sequences, maxlen=max_length_fr, padding='post')

    return eng_padded, fr_padded, eng_tokenizer, fr_tokenizer, max_length_eng, max_length_fr

# Load and preprocess the data
df = load_and_preprocess_data()

# Tokenize and pad the sequences
eng_padded, fr_padded, eng_tokenizer, fr_tokenizer, max_length_eng, max_length_fr = tokenize_and_pad(df)

# Build the GRU-based Seq2Seq model
vocab_size_eng = len(eng_tokenizer.word_index)
vocab_size_fr = len(fr_tokenizer.word_index)

model_new = Sequential()
model_new.add(Embedding(vocab_size_eng + 1, 100, input_length=max_length_eng))
model_new.add(GRU(20))
model_new.add(RepeatVector(int(max_length_fr))) # Convert max_length_fr to a Python integer
model_new.add(GRU(20, return_sequences=True))
model_new.add(TimeDistributed(Dense(vocab_size_fr + 1, activation="softmax")))

model_new.build(input_shape=(None, max_length_eng))
model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_new.summary()

# Train the model with EarlyStopping
french_output = np.expand_dims(fr_padded, -1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model_new.fit(
    eng_padded, french_output,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Save the model
model_new.save('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/gru_seq2seq_model.h5')

# Save the tokenizers
with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/eng_tokenizer.pkl', 'wb') as file:
    pickle.dump(eng_tokenizer, file)
with open('/Users/amnaalmurshda/Desktop/FLASK_Trans/models/fr_tokenizer.pkl', 'wb') as file:
    pickle.dump(fr_tokenizer, file)

print("Model and tokenizers saved successfully.")