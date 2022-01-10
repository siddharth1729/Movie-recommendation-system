#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/rposhala/Recommender-System-on-MovieLens-dataset/blob/main/Item_based_Collaborative_Recommender_System_using_KNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
print("libraries loaded....")


# In[2]:


DATASET_LINK='http://files.grouplens.org/datasets/movielens/ml-100k.zip'


# In[3]:


get_ipython().system('wget -nc http://files.grouplens.org/datasets/movielens/ml-100k.zip')
get_ipython().system('unzip -n ml-100k.zip')


# ## Loading MovieLens dataset

# Loading u.info     -- The number of users, items, and ratings in the u data set.

# In[4]:


overall_stats = pd.read_csv('ml-100k/u.info', header=None)
print("Details of users, items and ratings involved in the loaded movielens dataset: ",list(overall_stats[0]))


# Loading u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
# 
# ---
# 
# 
# 
#               Each user has rated at least 20 movies.  Users and items are
#               numbered consecutively from 1.  The data is randomly ordered. This is a tab separated list of 
# 	         user id | item id | rating | timestamp. 
#               The time stamps are unix seconds since 1/1/1970 UTC 

# In[5]:


## same item id is same as movie id, item id column is renamed as movie id
column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
dataset.head() 


# In[6]:


len(dataset), max(dataset['movie id']),min(dataset['movie id'])


# Loading u.item     -- Information about the items (movies); this is a tab separated
# 
#               list of
#               movie id | movie title | release date | video release date |
#               IMDb URL | unknown | Action | Adventure | Animation |
#               Children's | Comedy | Crime | Documentary | Drama | Fantasy |
#               Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
#               Thriller | War | Western |
#               The last 19 fields are the genres, a 1 indicates the movie
#               is of that genre, a 0 indicates it is not; movies can be in
#               several genres at once.
#               The movie ids are the ones used in the u.data data set.
# 

# In[7]:


d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
column_names2


# In[8]:


items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
items_dataset


# In[9]:


movie_dataset = items_dataset[['movie id','movie title']]
movie_dataset.head()


# Looking at length of original items_dataset and length of unique combination of rows in items_dataset after removing movie id column

# In[10]:


## looking at length of original items_dataset and length of unique combination of rows in items_dataset after removing movie id column
len(items_dataset.groupby(by=column_names2[1:])),len(items_dataset)


# We can see there are 18 extra movie id's for already mapped movie title and the same duplicate movie id is assigned to the user in the user-item dataset.

# ## Merging required datasets

# In[11]:


merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
merged_dataset.head()


# A dataset is created from the existing merged dataset by grouping the unique user id and movie title combination and the ratings by a user to the same movie in different instances (timestamps) are averaged and stored in the new dataset.

# Example of a multiple rating scenario by an user to a specific movie:

# In[12]:


merged_dataset[(merged_dataset['movie title'] == 'Chasing Amy (1997)') & (merged_dataset['user id'] == 894)]


# In[13]:


refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})

refined_dataset.head()


# ## Exploratory data analysis
# 
# *   Plot the counts of each rating
# *   Plot rating frequency of each movie

# **Plot the counts of each rating**
# 
# we first need to get the counts of each rating from ratings data

# In[14]:


# num_users = len(refined_dataset.rating.unique())
# num_items = len(refined_dataset.movieId.unique())
num_users = len(refined_dataset['user id'].value_counts())
num_items = len(refined_dataset['movie title'].value_counts())
print('Unique number of users in the dataset: {}'.format(num_users))
print('Unique number of movies in the dataset: {}'.format(num_items))


# In[15]:


rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])
rating_count_df


# In[16]:


ax = rating_count_df.reset_index().rename(columns={'index': 'rating score'}).plot('rating','count', 'bar',
    figsize=(12, 8),
    title='Count for Each Rating Score',
    fontsize=12)

ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")


# We can see that number of 1.5, 2.5, 3.5, 4.5 ratings by the users are comparitively negligible.

# Ratings for the movies not seen by a user is by default considered as 0. Lets calculate and add it to the existing dataframe.

# In[17]:


total_count = num_items * num_users
zero_count = total_count-refined_dataset.shape[0]
zero_count


# In[18]:


# append counts of zero rating to df_ratings_cnt
rating_count_df = rating_count_df.append(
    pd.DataFrame({'count': zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()
rating_count_df


# Number of times no rating was given (forged as 0 in this case) is a lot more than other ratings.

# So let's take log transform for count values and then we can plot them to compare

# In[19]:


# add log count
rating_count_df['log_count'] = np.log(rating_count_df['count'])
rating_count_df


# In[20]:


rating_count_df = rating_count_df.reset_index().rename(columns={'index': 'rating score'})
rating_count_df


# In[21]:


ax = rating_count_df.plot('rating score', 'log_count', 'bar', figsize=(12, 8),
    title='Count for Each Rating Score (in Log Scale)',
    logy=True,
    fontsize=12,)

ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")


# We have already observed from the before bar plot that ratings 3 and 4 are given in more numbers by the users. Even the above graph suggests the same.
# 
#  Take away from this plot is by the number of missing ratings, we can estimate the level of sparsity in the matrix we are going to form. 

# **Plot rating frequency of all movies**

# In[22]:


refined_dataset.head()


# In[23]:


# get rating frequency
movies_count_df = pd.DataFrame(refined_dataset.groupby('movie title').size(), columns=['count'])
movies_count_df.head()


# In[24]:


# plot rating frequency of all movies
ax = movies_count_df     .sort_values('count', ascending=False)     .reset_index(drop=True)     .plot(
        figsize=(12, 8),
        title='Rating Frequency of All Movies',
        fontsize=12
    )
ax.set_xlabel("movie Id")
ax.set_ylabel("number of ratings")


# **As the size of MovieLens dataset picked for this project is small. There is no need of removing rarely rated movies or users who has given rating for fewer movies.**
# 
# **Also because the dataset considered is small, we do not see the long-tail property which will be the scenario with the distribution of ratings.**
# 
# *If the dataset is larger, then* (this can be referred when we do similar kind of tasks with a larger dataset, just for future reference)
# 
# The distribution of ratings among movies often satisfies a property in real-world settings, which is referred to as the long-tail property. According to this property, only a small fraction of the items are rated frequently. Such items are referred to as popular items. The vast majority of items are rated rarely. This results in a highly skewed distribution of the underlying ratings.

# # Training KNN model to build item-based collaborative Recommender System.

# **Reshaping the dataframe**
# 
# We need to transform (reshape in this case) the data in such a way that each row of the dataframe represents a movie and each column represents a different user. So we want the data to be [movies, users] array if movie is the subject where similar movies must be found and [users, movies] array for reverse.
# 
# To reshape the dataframe, we will pivot the dataframe to the wide format with movies as rows and users as columns. As we know that not all users watch all the movies, we can expect a lot of missing values. We will have to fill those missing observations with 0s since we are going to perform linear algebra operations (calculating distances between vectors). 
# 
# Finally, we transform the values of the dataframe into a scipy sparse matrix for most efficient calculations.
# 
# This dataframe is then fed into a KNN model.

# ## Movie Recommendation using KNN with Input as **User id**, Number of similar users should the model pick and Number of movies you want to get recommended:

# 1. Reshaping model in such a way that each user has n-dimensional rating space where n is total number of movies
# 
#  We will train the KNN model inorder to find the closely matching similar users to the user we give as input and we recommend the top movies which would interest the input user.

# In[25]:


# pivot and create movie-user matrix
user_to_movie_df = refined_dataset.pivot(
    index='user id',
     columns='movie title',
      values='rating').fillna(0)

user_to_movie_df.head()


# In[26]:


# transform matrix to scipy sparse matrix
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
user_to_movie_sparse_df


# **Fitting K-Nearest Neighbours model to the scipy sparse matrix:**

# In[27]:


knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_movie_sparse_df)


# In[28]:


## function to find top n similar users of the given input user 
def get_similar_users(user, n = 5):
  ## input to this function is the user and number of top similar users you want.

  knn_input = np.asarray([user_to_movie_df.values[user-1]])  #.reshape(1,-1)
  # knn_input = user_to_movie_df.iloc[0,:].values.reshape(1,-1)
  distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n+1)
  
  print("Top",n,"users who are very much similar to the User-",user, "are: ")
  print(" ")
  for i in range(1,len(distances[0])):
    print(i,". User:", indices[0][i]+1, "separated by distance of",distances[0][i])
  return indices.flatten()[1:] + 1, distances.flatten()[1:]


# **Specify User id and Number of similar users we want to consider here**

# In[29]:


from pprint import pprint
user_id = 779
print(" Few of movies seen by the User:")
pprint(list(refined_dataset[refined_dataset['user id'] == user_id]['movie title'])[:10])
similar_user_list, distance_list = get_similar_users(user_id,5)


# **With the help of the KNN model built, we could get desired number of top similar users.**
# 
# **Now we will have to pick the top movies to recommend.**
# 
# **One way would be by taking the average of the existing ratings given by the similar users and picking the top 10 or 15 movies to recommend to our current user.**
# 
# **But I feel recommendation would be more effective if we define weights to ratings by each similar user based on the thier distance from the input user. Defining these weights would give us the accurate recommendations by eliminating the chance of decision manipulation by the users who are relatively very far from the input user.**

# In[30]:


similar_user_list, distance_list


# In[31]:


weightage_list = distance_list/np.sum(distance_list)
weightage_list


# Getting ratings of all movies by derived similar users

# In[32]:


mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
mov_rtngs_sim_users


# In[33]:


movies_list = user_to_movie_df.columns
movies_list


# In[34]:


print("Weightage list shape:", len(weightage_list))
print("mov_rtngs_sim_users shape:", mov_rtngs_sim_users.shape)
print("Number of movies:", len(movies_list))


# **Broadcasting weightage matrix to similar user rating matrix. so that it gets compatible for matrix operations**

# In[35]:


weightage_list = weightage_list[:,np.newaxis] + np.zeros(len(movies_list))
weightage_list.shape


# In[36]:


new_rating_matrix = weightage_list*mov_rtngs_sim_users
mean_rating_list = new_rating_matrix.sum(axis =0)
mean_rating_list


# In[37]:


from pprint import pprint
def recommend_movies(n):
  n = min(len(mean_rating_list),n)
  # print(np.argsort(mean_rating_list)[::-1][:n])
  pprint(list(movies_list[np.argsort(mean_rating_list)[::-1][:n]]))


# In[38]:


print("Movies recommended based on similar users are: ")
recommend_movies(10)


# It had been observed that, this recommendation system built can be made more efficient as it has few drawbacks.
# 
# **Drawbacks:**
# 
# **1.** But this recommendation system has a drawback, it also **recommends movies which are already seen by the given input User.**
# 
# **2.** And also there is a possibility of recommending the **movies which are not at all seen by any of the similar users.**

# **Above drawbacks are addressed and a new recommender system with modification is built**
# 
# Below function is defined to remove the movies which are already seen the current user and not at all seen by any of the similar users.

# In[39]:



def filtered_movie_recommendations(n):
  
  first_zero_index = np.where(mean_rating_list == 0)[0][-1]
  sortd_index = np.argsort(mean_rating_list)[::-1]
  sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
  n = min(len(sortd_index),n)
  movies_watched = list(refined_dataset[refined_dataset['user id'] == user_id]['movie title'])
  filtered_movie_list = list(movies_list[sortd_index])
  count = 0
  final_movie_list = []
  for i in filtered_movie_list:
    if i not in movies_watched:
      count+=1
      final_movie_list.append(i)
    if count == n:
      break
  if count == 0:
    print("There are no movies left which are not seen by the input users and seen by similar users. May be increasing the number of similar users who are to be considered may give a chance of suggesting an unseen good movie.")
  else:
    pprint(final_movie_list)


# In[40]:


filtered_movie_recommendations(10)


# Coding up all of the above individual cells into a function.
# 
# Giving Input as **User id, Number of similar Users to be considered, Number of top movie we want to recommend**

# In[41]:


from pprint import pprint

def recommender_system(user_id, n_similar_users, n_movies): #, user_to_movie_df, knn_model):
  
  print("Movie seen by the User:")
  pprint(list(refined_dataset[refined_dataset['user id'] == user_id]['movie title']))
  print("")

  # def get_similar_users(user, user_to_movie_df, knn_model, n = 5):
  def get_similar_users(user, n = 5):
    
    knn_input = np.asarray([user_to_movie_df.values[user-1]])
    
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n+1)
    
    print("Top",n,"users who are very much similar to the User-",user, "are: ")
    print(" ")

    for i in range(1,len(distances[0])):
      print(i,". User:", indices[0][i]+1, "separated by distance of",distances[0][i])
    print("")
    return indices.flatten()[1:] + 1, distances.flatten()[1:]


  def filtered_movie_recommendations(n = 10):
  
    first_zero_index = np.where(mean_rating_list == 0)[0][-1]
    sortd_index = np.argsort(mean_rating_list)[::-1]
    sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
    n = min(len(sortd_index),n)
    movies_watched = list(refined_dataset[refined_dataset['user id'] == user_id]['movie title'])
    filtered_movie_list = list(movies_list[sortd_index])
    count = 0
    final_movie_list = []
    for i in filtered_movie_list:
      if i not in movies_watched:
        count+=1
        final_movie_list.append(i)
      if count == n:
        break
    if count == 0:
      print("There are no movies left which are not seen by the input users and seen by similar users. May be increasing the number of similar users who are to be considered may give a chance of suggesting an unseen good movie.")
    else:
      pprint(final_movie_list)

  similar_user_list, distance_list = get_similar_users(user_id,n_similar_users)
  weightage_list = distance_list/np.sum(distance_list)
  mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
  movies_list = user_to_movie_df.columns
  weightage_list = weightage_list[:,np.newaxis] + np.zeros(len(movies_list))
  new_rating_matrix = weightage_list*mov_rtngs_sim_users
  mean_rating_list = new_rating_matrix.sum(axis =0)
  print("")
  print("Movies recommended based on similar users are: ")
  print("")
  filtered_movie_recommendations(n_movies)


# In[42]:


print("Enter user id")
user_id= int(input())
print("number of similar users to be considered")
sim_users = int(input())
print("Enter number of movies to be recommended:")
n_movies = int(input())
recommender_system(user_id,sim_users,n_movies)
# recommender_system(300, 15,15)


# ## Movie Recommendation using KNN with Input as **Movie Name** and Number of movies you want to get recommended:

# 2. Reshaping model in such a way that each movie has n-dimensional rating space where n is total number of users who could rate.
# 
#  We will train the KNN model inorder to find the closely matching similar movies to the movie we give as input and we recommend the top movies which would more closely align to the movie we have given.

# In[43]:


# pivot and create movie-user matrix
movie_to_user_df = refined_dataset.pivot(
     index='movie title',
   columns='user id',
      values='rating').fillna(0)

movie_to_user_df.head()


# In[44]:


# transform matrix to scipy sparse matrix
movie_to_user_sparse_df = csr_matrix(movie_to_user_df.values)
movie_to_user_sparse_df


# Extracting movie names into a list:

# In[45]:


movies_list = list(movie_to_user_df.index)
movies_list[:10]


# Creating a dictionary with movie name as key and its index from the list as value:

# In[46]:


movie_dict = {movie : index for index, movie in enumerate(movies_list)}
print(movie_dict)


# In[47]:


case_insensitive_movies_list = [i.lower() for i in movies_list]


# Fitting a KNN model:

# In[48]:


knn_movie_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_movie_model.fit(movie_to_user_sparse_df)


# In[49]:


## function to find top n similar users of the given input user 
def get_similar_movies(movie, n = 10):
  ## input to this function is the movie and number of top similar movies you want.
  index = movie_dict[movie]
  knn_input = np.asarray([movie_to_user_df.values[index]])
  n = min(len(movies_list)-1,n)
  distances, indices = knn_movie_model.kneighbors(knn_input, n_neighbors=n+1)
  
  print("Top",n,"movies which are very much similar to the Movie-",movie, "are: ")
  print(" ")
  for i in range(1,len(distances[0])):
    print(movies_list[indices[0][i]])
  


# Testing the recommender system with basic input with the movie names
# 

# In[50]:


from pprint import pprint
movie_name = '101 Dalmatians (1996)'

get_similar_movies(movie_name,15)


# **Dynamically suggesting** movie name from the existing movie corpus we have, based on the user input using try and except architecture.

# Defining a function which outputs movie names as suggestion when the user mis spells the movie name. **User might have intended to type any of these movie names.**

# In[51]:


# function which takes input and returns suggestions for the user

def get_possible_movies(movie):

    temp = ''
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie :
      out = []
      temp += i
      for j in possible_movies:
        if temp in j:
          out.append(j)
      if len(out) == 0:
          return possible_movies
      out.sort()
      possible_movies = out.copy()

    return possible_movies


# This function provides user with **movie name suggestions if movie name is mis-spelled** or **Recommends similar movies to the input movie** if the movie name is valid.

# In[52]:


class invalid(Exception):
    pass

def spell_correction():
    
    try:

      movie_name = input("Enter the Movie name: ")
      movie_name_lower = movie_name.lower()
      if movie_name_lower not in case_insensitive_movies_list :
        raise invalid
      else :
        # movies_list[case_insensitive_country_names.index(movie_name_lower)]
        num_recom = int(input("Enter Number of movie recommendations needed: "))
        get_similar_movies(movies_list[case_insensitive_movies_list.index(movie_name_lower)],num_recom)

    except invalid:

      possible_movies = get_possible_movies(movie_name_lower)

      if len(possible_movies) == len(movies_list) :
        print("Movie name entered is does not exist in the list ")
      else :
        indices = [case_insensitive_movies_list.index(i) for i in possible_movies]
        print("Entered Movie name is not matching with any movie from the dataset . Please check the below suggestions :\n",[movies_list[i] for i in indices])
        spell_correction()


# In[53]:


spell_correction()


# Observation on above built KNN Recommender System:
# 
# An interesting observation would be that the above KNN model for movies recommends movies that are produced in very similar years of the input movie. However, the cosine distance of all those recommendations are observed to be actually quite small. This might be because there are too many zero values in our movie-user matrix. With too many zero values in our data, the data sparsity becomes a real issue for KNN model and the distance in KNN model starts to fall apart. So I'd like to dig deeper and look closer inside our data.

# 
# Let's now look at how sparse the movie-user matrix is by calculating percentage of zero values in the data.

# In[54]:


# calcuate total number of entries in the movie-user matrix
num_entries = movie_to_user_df.shape[0] * movie_to_user_df.shape[1]
# calculate total number of entries with zero values
num_zeros = (movie_to_user_df==0).sum(axis=1).sum()
# calculate ratio of number of zeros to number of entries
ratio_zeros = num_zeros / num_entries
print('There is about {:.2%} of ratings in our data is missing'.format(ratio_zeros))


# This result confirms the above hypothesis. The vast majority of entries in our data is zero. This explains why the distance between similar items or opposite items are both pretty large.
# 
# So, lets try out deep learning models and Natural Language Processing techniques in the next segment of this project.

# In[ ]:





# # Rough Work

# In[55]:


d = np.asarray([2,4,4,8,2])
d = d/np.sum(d)
d


# In[56]:


e = np.asarray([[10,20,30],[30,40,20],[50,30,10],[40,30,10],[30,20,50]])
e


# In[57]:


x = np.arange(4)
xx = x.reshape(4,1)
y = np.ones(5)
xx+y, y.shape, xx.shape


# In[58]:


d = d[:,np.newaxis] + np.zeros(3)
d


# In[59]:


f = d * e
f


# In[ ]:


f.sum(axis=0)


# In[ ]:


q = e.mean(axis=0)
q


# In[ ]:


l = np.asarray([3,7,1,2,8,9,3])
np.argsort(l)[::-1][:4]


# In[ ]:


p = np.asarray([3,6,2,8,1,0,6,2,6,0,7,9,5,0])
np.argsort(p)[::-1], np.sort(p)[::-1]


# In[ ]:


# list(p)[::-1].index(0)
j = np.where(p == 0)[0][-1]
j


# In[ ]:


j = np.where(p == 0)[0][-1]
h = np.argsort(p)[::-1]
h[:list(h).index(j)]


# In[ ]:


f = 'rohith kumar 346'
i = ['hit','ro','rohi','34',' ','itho','ohit']

for j in i:
  if j in f:
    print(j)


# In[ ]:


d = [2,3,76,4,7,1]
e = [3,1,7,2]
[d.index(i) for i in e]


# In[ ]:





# In[ ]:





# In[ ]:




