from flask import Flask, render_template,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

#load the dataset
df=pd.read_csv('movie_metadata.csv')
df.drop_duplicates(subset="movie_title",keep=False,inplace=True)
df['ID']=np.arange(len(df))
important_features=['ID','movie_title','director_name','genres','actor_1_name','actor_2_name','actor_3_name']
df=df[important_features]

#preprocesing  of data
for x in important_features:
    df[x]=df[x].fillna(' ')    
df['movie_title']=df['movie_title'].apply(lambda x:x.replace(u'\xa0',u''))
df['movie_title']=df['movie_title'].apply(lambda x:x.strip())

#function that combine director_name,genres,actor and return them as string
def combine(row):
        return row['director_name']+" "+row["genres"]+" "+row['actor_1_name']+" "+row['actor_2_name']+" "+row['actor_3_name']
df["combined"]=df.apply(combine,axis=1)

#Convert a collection of text documents to a matrix of token counts
cv=CountVectorizer()
count=cv.fit_transform(df["combined"])

#finding the  similarity
cosine_simi=cosine_similarity(count)
def get_tittle(Id):
    return df[df.ID==Id]["movie_title"].values[0]

def cont_recommend(user_liking,n):
    movie_index=get_id(user_liking)
    similar_movies=list(enumerate(cosine_simi[movie_index]))
    sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)
    sorted_similar_movies
    i=0
    j=0
    l=[1]*n
    for movie in sorted_similar_movies:    
        x=get_tittle(movie[0])
        if i==0:
            i=i+1
        else:    
            l[j]=x
            j=j+1
            i=i+1
        if i>n:
            break
    return l

def word2vec(word):
    from collections import Counter
    from math import sqrt
    cn= Counter(word)
    # precomputes a set of the different characters
    s = set(cn)
    # precomputes the "length" of the word vector
    l = sqrt(sum(c*c for c in cn.values()))

    # return a tuple
    return cn, s, l

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]  
# retuns id    

def get_id(tittle):
    if tittle in df.movie_title.unique():
        return  df[df.movie_title==tittle]['ID'].values[0]
    else:
        return -1

def notfound(s1,s2):
    low=s1.lower()
    sim=-1
    print(sim)
    for i in range(len(s2)):
        low1=s2[i].lower()
        m=word2vec(low)
        n=word2vec(low1)
        c=cosdis(m,n)
        if(c>sim):
            sim=c
            a=s2[i]
    return a     
        
app = Flask(__name__)

@app.route('/')
#home page
def home():
   return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
#output page
def predict():
    flag=0
    if request.method=='GET':
        user_liking=request.args.get('movie')
    else:
        user_liking=request.form['movie']
    l1=list(df['movie_title'])
    if user_liking in l1:
        output=cont_recommend(user_liking,10)
        output=list(output)
    else:
        flag=1
        movie=notfound(user_liking,l1)   
        output=cont_recommend(movie,10)
    if flag==1:
        return render_template('not_found.html',output=output,y=movie)
    else:     
        return render_template('after.html',output=output)
if __name__ == '__main__':
   app.run(debug=True)