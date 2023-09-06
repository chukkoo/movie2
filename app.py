#item2vecの因子数とかエポック数を変えてみる
import streamlit as st
import pandas as pd
import numpy as np
import gensim
import random

st.title('映画レコメンド')
click = st.button('ランダム映画を表示')

# 映画情報の読み込み
movies = pd.read_csv("data/movies.tsv", sep="\t")

# 学習済みのitem2vecモデルの読み込み
model = gensim.models.word2vec.Word2Vec.load("data/item2vec.model")


# 映画IDとタイトルを辞書型に変換
movie_titles = movies["title"].tolist()
movie_ids = movies["movie_id"].tolist()
movie_genres = movies["genre"].tolist()
movie_tags = movies["tag"].tolist()
movie_id_to_title = dict(zip(movie_ids, movie_titles)) #辞書型で
movie_title_to_id = dict(zip(movie_titles, movie_ids))
movie_id_to_genre = dict(zip(movie_ids, movie_genres))
movie_id_to_tag = dict(zip(movie_ids, movie_tags))

#アルファベット順にソート
#sort_alphabet = st.button("アルファベット順にソート")

#if sort_alphabet:
#    movie_titles.sort()
#特殊文字がある？ためうまくできなかった

if click:
    random_movie_id = random.choice(movie_ids)
    random_movie_title = movie_id_to_title[random_movie_id]
    st.write(f"ランダムに選ばれた映画は{random_movie_title} (id={random_movie_id})です")


#検索機能
search = st.text_input("映画を検索してください", "")
if search:

    filtered_movies = [title for title in movie_titles if search.lower() in title.lower()]
else:
    filtered_movies = movie_titles

selected_movie = st.selectbox("映画を選んでください", filtered_movies)

st.markdown("## 1本の映画に対して似ている映画を表示する")
#selected_movie = st.selectbox("映画を選んでください", movie_titles)
selected_movie_id = movie_title_to_id[selected_movie]
st.write(f"あなたが選択した映画は{selected_movie}(id={selected_movie_id})です")

# 似ている映画を表示
st.markdown(f"### {selected_movie}に似ている映画")

#何番目まで表示するかを自分で選べるように
num_similar_movies = st.slider("上から何番目までの似ている映画を表示しますか？選んでください", min_value=1, max_value=20, value=5)

results = []
for movie_id, score in model.wv.most_similar(selected_movie_id,topn=num_similar_movies): #topnで上から何番目
    title = movie_id_to_title[movie_id]
    genre = movie_id_to_genre[movie_id]
    tag = movie_id_to_tag[movie_id]
    results.append({"movie_id":movie_id, "title": title, "score": score,"genre":genre,"tag":tag}) 
results = pd.DataFrame(results)
st.write(results)

#検索機能2
search2 = st.text_input("映画を検索してください2", "")
if search2:

    filtered_movies = [title for title in movie_titles if search2.lower() in title.lower()]
else:
    filtered_movies = movie_titles

selected_movies = st.multiselect("映画を複数選んでください", filtered_movies)

st.markdown("## 複数の映画を選んでおすすめの映画を表示する")

#何番目まで表示するかを自分で選べるように2
num_similar_movies2 = st.slider("上から何番目までの似ている映画を表示しますか？選んでください2", min_value=1, max_value=20, value=5)

#selected_movies = st.multiselect("映画を複数選んでください", movie_titles) #複数だからmultiselect
selected_movie_ids = [movie_title_to_id[movie] for movie in selected_movies] #idに変換
vectors = [model.wv.get_vector(movie_id) for movie_id in selected_movie_ids] #映画のベクトルを取得
if len(selected_movies) > 0:
    user_vector = np.mean(vectors, axis=0)
    st.markdown(f"### おすすめの映画")
    recommend_results = []
    for movie_id, score in model.wv.most_similar(user_vector,topn=num_similar_movies2):
        title = movie_id_to_title[movie_id]
        genre = movie_id_to_genre[movie_id]
        tag = movie_id_to_tag[movie_id]
        recommend_results.append({"movie_id":movie_id, "title": title, "score": score,"genre":genre,"tag":tag})
    recommend_results = pd.DataFrame(recommend_results)
    st.write(recommend_results)
