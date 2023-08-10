import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

train_data = pd.read_csv('train.csv', sep=':::', header=None, engine='python', encoding='MacRoman')
train_data.columns = ['id', 'title', 'genre', 'plot']
X_train = train_data['plot']
y_train = train_data['genre']
test_data = pd.read_csv('test.csv', sep=':::', header=None, engine='python', encoding='MacRoman')
test_data.columns = ['id', 'title', 'plot']
X_test = test_data['plot']
solution_data = pd.read_csv('test_solution.csv', sep=':::', header=None, engine='python', encoding='MacRoman')
solution_data.columns = ['id', 'title', 'genre', 'plot']
y_test = solution_data['genre']

genre_to_int = {genre: i for i, genre in enumerate(y_train.unique())}
int_to_genre = {i: genre for genre, i in genre_to_int.items()}
y_train = y_train.map(genre_to_int)
y_test = y_test.map(genre_to_int)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_sequences, maxlen=500)
X_test_padded = pad_sequences(X_test_sequences, maxlen=500)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(genre_to_int), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train, epochs=3, batch_size=32, validation_split=0.2)

y_pred_probs = model.predict(X_test_padded)
y_pred_labels = y_pred_probs.argmax(axis=-1)
y_pred_genre = [int_to_genre[i] for i in y_pred_labels] 

output_df = pd.DataFrame({
    'id': test_data['id'],
    'title': test_data['title'],
    'predicted_genre': y_pred_genre
})

output_df.to_csv('predictions.csv', index=False)

print("Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=list(genre_to_int.keys())))
