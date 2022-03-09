import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

scoredata = pd.read_csv('./data/rating.csv')
moviedata = pd.read_csv('./data/movie.csv')

scoredata = scoredata[:20000]
moviedata = moviedata[:10000]

scoredata = scoredata.drop(['timestamp'],axis='columns')
result = pd.merge(moviedata, scoredata, on="movieId", how="right")
print(result)

scoretable2= result.pivot(['movieId','title'],'userId','rating')


scoretable = pd.pivot_table(
    result,
    columns='userId',
    index = ['movieId','title'],
    values='rating',
    fill_value = 0
)

print(scoretable)
print(scoretable2)

sim = cosine_similarity(scoredata)
#sim2 = cosine_similarity(scoretable2)

print(sim)
#print(sim2)