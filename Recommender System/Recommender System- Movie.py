#!/usr/bin/env python
# coding: utf-8

# # Recommender Statement

# # Problem Statement

# - This notebook implements a movie recommender system. 
# - Recommender systems are used to suggest movies or songs to users based on their interest or usage history. 
# - For example, Netflix recommends movies to watch based on the previous movies you've watched.  
# - In this example, we will use Item-based Collaborative Filter 

# ## Step 1: Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Step 2: Import Dataset

# In[2]:


movie_titles_df=pd.read_csv("Movie_Id_Titles")
movie_titles_df.head()


# In[4]:


movie_ratings_df=pd.read_csv("u.data", sep='\t',names=['user_id', 'item_id', 'ratings', 'timestamp'])
movie_ratings_df.head()


# In[6]:


movie_ratings_df=movie_ratings_df.drop(['timestamp'],axis=1)


# In[7]:


movie_ratings_df.head()


# In[9]:


movie_ratings_df.describe()


# In[10]:


movie_ratings_df.info()


# ## Merge two datasets to get title names

# In[12]:


movies_ratings_df=pd.merge(movie_ratings_df,movie_titles_df, on='item_id')


# In[13]:


movies_ratings_df.head()


# In[15]:


movies_ratings_df.shape


# ## Step 3: Visualize dataset

# In[17]:


movies_ratings_df.groupby('title')['ratings'].describe()


# In[19]:


ratings_df_mean=movies_ratings_df.groupby('title')['ratings'].describe()['mean']


# In[20]:


ratings_df_mean.head()


# In[21]:


ratings_df_count=movies_ratings_df.groupby('title')['ratings'].describe()['count']


# In[22]:


ratings_df_count.head()


# In[23]:


ratings_mean_count_df=pd.concat([ratings_df_mean, ratings_df_count], axis=1)


# In[24]:


ratings_mean_count_df.head()


# In[25]:


ratings_mean_count_df.reset_index()


# In[26]:


ratings_mean_count_df['mean'].plot(bins=100,kind='hist', color='r')


# In[27]:


ratings_mean_count_df['count'].plot(bins=100,kind='hist', color='r')


# In[28]:


# lets see high rated movies
ratings_mean_count_df[ratings_mean_count_df['mean']==5]


# In[29]:


ratings_mean_count_df.sort_values('count', ascending=False).head()


# ## Step 4: Perform Item based collaborative filtering on one movie

# In[30]:


userid_movietitle_matrix=movies_ratings_df.pivot_table(index='user_id', columns='title', values='ratings')


# In[31]:


userid_movietitle_matrix


# In[32]:


# lets have sample movie as Titanic and based on item ratings recommend other movies
titanic=userid_movietitle_matrix['Titanic (1997)']


# In[33]:


titanic


# In[34]:


#lets calculate correlations
titanic_correlations=pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations= titanic_correlations.join(ratings_mean_count_df)


# In[35]:


titanic_correlations


# In[36]:


titanic_correlations.dropna(inplace=True)


# In[37]:


titanic_correlations


# In[38]:


titanic_correlations.sort_values('Correlation', ascending=False)


# In[41]:


titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation',ascending=False).head()


# ## Step 5: Create an Item based collaborative filter on entire dataset

# In[42]:


userid_movietitle_matrix


# In[43]:


# Pearson is standard correlation coefficient having values as -1,0,1 where 0 means not linearly related
movie_correlations=userid_movietitle_matrix.corr(method='pearson', min_periods=80)


# In[44]:


movie_correlations


# In[46]:


# we have another dataframe with our ratings. lets feed this data in recommender system to see output.
myRatings=pd.read_csv('My_ratings.csv')


# In[47]:


myRatings


# In[48]:


len(myRatings.index)


# In[49]:


myRatings['Movie Name'][0]


# In[50]:


similar_movie_list=pd.Series()
for i in range(0,2):
    similar_movie=movie_correlations[myRatings['Movie Name'][i]].dropna()
    similar_movie=similar_movie.map(lambda x: x*myRatings['Ratings'][i])
    similar_movie_list=similar_movie_list.append(similar_movie)


# In[51]:


similar_movie_list.sort_values(inplace=True, ascending=False)
print(similar_movie_list.head())

