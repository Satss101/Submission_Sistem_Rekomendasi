import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('filmtv_movies.csv')
# Display the first few rows of the dataset
df.head(5)

# cek baris dan kolom
df.shape

#cek info
df.info()

#cek missing value
plt.hist(df.avg_vote, color='#B4E1FF', edgecolor='black')
plt.ylabel('Total')
plt.xlabel('avg_vote')
plt.title("Movie's avg_vote Distribution")
plt.show()

# Visualisasi jumlah film pada setiap genre
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='genre', hue='genre', order=df['genre'].value_counts().index, palette='viridis', legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('Jumlah Film per Genre')
plt.xlabel('Genre')
plt.ylabel('Jumlah Film')
plt.tight_layout()
plt.show()

# print describe df
df.describe().T

# cek duplikat
df.duplicated().sum()

# cek nilai kosong
df.isnull().sum()

# mengambil beberapa feature saja
df_set = df[['title', 'avg_vote', 'filmtv_id', 'genre', 'directors','description']].copy()

# cek duplikat pada df_set
df_set.isnull().sum()

# drop missing values
df_set.dropna(inplace=True)
print(df_set.shape)

# cek total missing values after dropping
df_set.isnull().sum()

# cek genre unik
df_set.genre.unique()

# cek directors unik
df_set.directors.unique().tolist()[:10]

# Menghitung jumlah baris dengan lebih dari satu director (ditandai dengan koma)
multi_director_count = df_set['directors'].str.contains(',', na=False).sum()
print(f"Jumlah baris dengan lebih dari satu director: {multi_director_count}")

# Menghapus baris dengan lebih dari satu director (ada koma di kolom 'directors')
df_set = df_set[~df_set['directors'].str.contains(',', na=False)]
print(df_set.shape)

# Membuat DataFrame baru untuk Content-Based Filtering (CBF)
df_cbf = pd.DataFrame({
    'filmtv_id': df_set['filmtv_id'],
    'title': df_set['title'],
    'genre': df_set['genre'],
    'description': df_set['description'].apply(lambda x: x.lower().strip())
})
# Menampilkan DataFrame CBF
df_cbf.head(5)

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
 
# Melakukan perhitungan idf pada data cuisine
tf.fit(df_cbf['genre']) 
 
# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(df_cbf['genre'])
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

# Menampilkan DataFrame dengan fitur tf-idf
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=df_cbf.title
).sample(28, axis=1).sample(10, axis=0)

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama resto
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_cbf['title'], columns=df_cbf['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix pada setiap resto
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

def title_recommendations(title, similarity_data=cosine_sim_df, items=df_cbf[['title', 'genre','description']], k=5):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop nama_resto agar nama resto yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(title, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

df_cbf[df_cbf.title.eq('The Adventures of Don Juan')]

# Mendapatkan rekomendasi restoran yang mirip dengan KFC
title_recommendations('The Adventures of Don Juan')

# importing necessary libraries for deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# copy dataframe untuk cross validation
df_cv = df_set.copy()
df_cv.head(5)

# Import library
import string
import numpy as np
# Membuat list huruf a-z
letters = list(string.ascii_lowercase)
n_letters = len(letters)
n_rows = len(df_cv)
# Hitung jumlah baris per huruf (porsi merata)
rows_per_letter = n_rows // n_letters
remainder = n_rows % n_letters
# Buat list huruf untuk semua baris
letter_list = []
for i, letter in enumerate(letters):
    count = rows_per_letter + (1 if i < remainder else 0)
    letter_list.extend([letter] * count)
# Acak urutan huruf
np.random.shuffle(letter_list)
# Tambahkan ke dataframe sebagai kolom baru
df_cv['user'] = letter_list
df_cv.head()

# Visualisasi distribusi user a-z pada df_cv
plt.figure(figsize=(10,5))
sns.countplot(data=df_cv, x='user', hue='user', order=sorted(df_cv['user'].unique()), palette='tab20', legend=False)
plt.title('Distribusi user a-z pada df_cv')
plt.xlabel('Random Letter')
plt.ylabel('Jumlah')
plt.show()

# Membuat salinan dari df_cv untuk cross validation
df_cv_first = df_cv.copy()
# Mengubah filmtv_id menjadi list tanpa nilai yang sama
user_id = df_cv['user'].unique().tolist()
print('list user: ', user_id)
 
# Melakukan encoding filmtv_id
user_to_user_encoded = {x: i for i, x in enumerate(user_id)}
print('encoded filmtv_id : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke ke filmtv_id
user_encoded_to_user = {i: x for i, x in enumerate(user_id)}
print('encoded angka ke filmtv_id: ', user_encoded_to_user)

# Mengubah title menjadi list tanpa nilai yang sama
title_ids = df_cv['title'].unique().tolist()
print('list title: ', title_ids)
 
# Melakukan encoding title
title_to_title_encoded = {x: i for i, x in enumerate(title_ids)}
print('encoded title : ', title_to_title_encoded)
 
# Melakukan proses encoding angka ke ke title
title_encoded_to_title = {i: x for i, x in enumerate(title_ids)}
print('encoded angka ke title: ', title_encoded_to_title)

# tampilkan df_cv dengan kolom user dan title yang sudah di-encode
df_cv.head(5)

# Mapping filmtv_id ke dataframe user
df_cv['user'] = df_cv['user'].map(user_to_user_encoded)
 
# Mapping title ke dataframe resto
df_cv['title'] = df_cv['title'].map(title_to_title_encoded)

# Mengacak dataset
df_cv = df_cv.sample(frac=1, random_state=42)
# tampilkan 5 baris pertama dari df_cv
df_cv.head(5)

# medapatkan jumlah filmtv_id
jumlah_user = len(user_to_user_encoded)
print('Jumlah filmtv_id: ', jumlah_user)
# mendapatkan jumlah title
jumlah_title = len(title_to_title_encoded)
print('Jumlah title: ', jumlah_title)
# nilai minimum vote
min_vote = df_cv['avg_vote'].min()
# nilai maksimum vote
max_vote = df_cv['avg_vote'].max()
# menampilkan informasi jumlah filmtv_id, jumlah title, nilai minimum vote, dan nilai maksimum vote
print('banyak filmtv_id: {}, banyak title: {}, nilai minimum vote: {}, nilai maksimum vote: {}'.format(
      jumlah_user, jumlah_title, min_vote, max_vote
))

# Membuat variabel x untuk mencocokkan data user dan resto menjadi satu value
x = df_cv[['user', 'title']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = df_cv['avg_vote'].apply(lambda x: (x - min_vote) / (max_vote - min_vote)).values
 
# Membagi menjadi 90% data train dan 10% data validasi
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
# Menampilkan bentuk dari x_train, x_val, y_train, dan y_val
print(x, y)

class RecommenderNet(tf.keras.Model):
  
 # Insialisasi fungsi
  def __init__(self, num_users, num_resto, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_resto = num_resto
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.resto_embedding = layers.Embedding( # layer embeddings resto
        num_resto,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_resto, 1) # layer embedding resto bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    resto_vector = self.resto_embedding(inputs[:, 1]) # memanggil layer embedding 3
    resto_bias = self.resto_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2) 
 
    x = dot_user_resto + user_bias + resto_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# Inisialisasi model
model = RecommenderNet(jumlah_user, jumlah_title, 50) # inisialisasi model
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# menggunakan callback
class callback_model(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('root_mean_squared_error')<0.06 and logs.get('val_root_mean_squared_error')<0.15):
            print("\n\nTarget tercapai\n")
            self.model.stop_training = True

# Melatih model
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 128,
    epochs = 50, 
    validation_data = (x_val, y_val),
    callbacks=[callback_model()]
)

# Plotting the training and validation loss
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
# Pilih user acak (dalam bentuk string, bukan integer)
user_id = df_cv_first.user.sample(1).iloc[0]
user_watched_movie = df_cv_first[df_cv_first.user == user_id]
# Mendapatkan daftar judul film yang sudah ditonton user_id
watched_titles = set(user_watched_movie['title'])
# Mendapatkan semua judul film yang tersedia
all_titles = set(df_set['title'])
# Mendapatkan judul film yang belum ditonton user_id
movie_not_watched = all_titles - watched_titles
print('Jumlah film yang belum ditonton oleh user {}: {}'.format(user_id, len(movie_not_watched)))
# Mengubah judul film yang belum ditonton menjadi encoded format
movie_not_watched = [[title_to_title_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)


# Melakukan prediksi rating untuk film yang belum ditonton oleh user_id
ratings = model.predict(user_movie_array).flatten()
 
# Mengurutkan rating dan mengambil 10 film dengan rating tertinggi
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    title_encoded_to_title.get(movie_not_watched[x][0]) for x in top_ratings_indices
]
# Menampilkan rekomendasi film untuk user_id
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Movie with high ratings from user')
print('----' * 8)
 
top_resto_user = (
    user_watched_movie.sort_values(
        by = 'avg_vote',
        ascending=False
    )
    .head(5)
    .title.values
)
# Menampilkan judul film yang belum ditonton oleh user_id beserta genre-nya
for movie in top_resto_user:
    for genre in df_cv_first[df_cv_first.title == movie].genre:
        print(movie, ':', genre)
        break
 
print('----' * 8)
print('Top 10 movie recommendation')
print('----' * 8)
# Menampilkan rekomendasi film yang belum ditonton oleh user_id beserta genre-nya
for movie in recommended_movie_ids:
    for genre in df_cv_first[df_cv_first.title == movie].genre:
        print(movie, ':', genre)
        break