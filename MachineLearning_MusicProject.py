#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv('D:/Pycharm-projects/music_genre.csv')
df


# In[50]:


df = df.drop_duplicates(['instance_id']) #we removed duplicated values


# In[51]:


df = df.drop(df.index[df['duration_ms'] <= 1.0])
df = df.drop('track_name', axis='columns')
df


# In[52]:


d = df.dropna(axis = 0, how ='any')  #we removed 'any' values that we are not going to use
df = df.drop(df.index[df['tempo'] == "?"])


# In[53]:


from sklearn.preprocessing import LabelEncoder
le_artists = LabelEncoder()
le_key = LabelEncoder()
le_musicGenre = LabelEncoder()
le_mode = LabelEncoder()
le_obtained = LabelEncoder()


# In[54]:


df['instance_id_n'] = le_artists.fit_transform(df['instance_id'])
df['artist_name_n'] = le_artists.fit_transform(df['artist_name'])
df['popularity_n'] = le_artists.fit_transform(df['popularity'])
df['danceability_n'] = le_artists.fit_transform(df['danceability'])
df['duration_ms_n'] = le_artists.fit_transform(df['duration_ms'])
df['energy_n'] = le_artists.fit_transform(df['energy'])
df['instrumentalness_n'] = le_artists.fit_transform(df['instrumentalness'])
df['key_n'] = le_artists.fit_transform(df['key'])
df['tempo_n'] = le_artists.fit_transform(df['tempo'])
df['obtained_n'] = le_artists.fit_transform(df['obtained_date'])
df['valence_n'] = le_artists.fit_transform(df['valence'])
df['music_genre_n'] = le_artists.fit_transform(df['music_genre'])
df['mode_n'] = le_artists.fit_transform(df['mode'])
df['tempo_n'] = le_artists.fit_transform(df['tempo'])
df['loudness_n'] = le_artists.fit_transform(df['loudness'])
df['liveness_n'] = le_artists.fit_transform(df['liveness'])
df['speechiness_n'] = le_artists.fit_transform(df['speechiness'])
df['acousticness_n'] = le_artists.fit_transform(df['acousticness'])


df


# In[55]:


df = df.drop(['artist_name', 'key', 'music_genre', 'mode', 'obtained_date','tempo','instance_id','popularity','acousticness'], axis='columns')
df = df.drop(['danceability', 'duration_ms', 'energy', 'instrumentalness', 'valence','loudness','liveness','speechiness'], axis='columns')


# In[56]:


df = df.sample(frac=1)
df


# In[57]:


sns.pairplot(df, x_vars= ['energy_n'], y_vars=['popularity_n'], height=20,aspect=2.0,kind='scatter')


# In[58]:


sns.pairplot(df, x_vars= ['music_genre_n'], y_vars=['popularity_n'], height=20,aspect=2.0,kind='scatter')


# In[59]:


fig = plt.figure()
ax =fig.add_subplot(111, projection='3d')
ax.scatter(df['instance_id_n'], df['acousticness_n'], df['energy_n'], c='r', marker='o',s=1)
plt.show()


# In[60]:


sns.pairplot(df, x_vars= ['popularity_n'], y_vars=['loudness_n'], height=20,aspect=2.0,kind='scatter')


# In[61]:


for_test = df.copy()


# In[62]:


training_data = for_test.sample(frac=0.8, random_state=25)
testing_data = for_test.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[63]:


print(training_data)


# In[64]:


training_data


# In[65]:


testing_data


# In[66]:


plt.scatter(training_data['popularity_n'], training_data['energy_n'], linewidths=1000, s=1)


# In[67]:


plt.scatter(testing_data['popularity_n'], testing_data['energy_n'], linewidths=1000, s=1)


# In[68]:


################## 


# In[69]:


plt.scatter(training_data['popularity_n'], training_data['artist_name_n'], linewidths=1000, s=1)


# In[70]:


plt.scatter(testing_data['popularity_n'], testing_data['artist_name_n'], linewidths=1000, s=1)


# In[71]:


############################


# In[72]:


plt.scatter(training_data['popularity_n'], training_data['music_genre_n'], linewidths=1000, s=1)


# In[73]:


plt.scatter(testing_data['popularity_n'], testing_data['music_genre_n'], linewidths=1000, s=1)


# In[74]:


###################################


# In[75]:


x = df[['popularity_n','tempo_n']]
y = df['obtained_n']


# In[76]:


x


# In[77]:


y


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = 0.3)


# In[39]:


x_train


# In[40]:


x_test


# In[41]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)


# In[42]:


clf.predict(x_test)


# In[43]:


clf.score(x_test, y_test) # %01 


# In[78]:


from numpy.random import default_rng
arr_indices_top_drop = default_rng().choice(df.index, size=25000, replace=False)
dff = df.drop(index=arr_indices_top_drop)
dff 


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(dff[['music_genre_n']],dff.artist_name_n, test_size =0.3)


# In[46]:


##################


# In[48]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0).fit(x_train, y_train)


# In[ ]:


model.predict(x_test)


# In[ ]:


##############


# In[79]:


newtry= dff.copy()
inputs = newtry.drop('popularity_n', axis='columns')
target = newtry['popularity_n']


# In[80]:


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs, target)


# In[81]:


model.score(inputs,target)


# In[82]:


##################


# In[83]:


df.info()


# In[84]:


###############


# In[85]:


from sklearn.svm import SVC
x_train, x_test, y_train, y_test = train_test_split(dff[['music_genre_n']],dff.artist_name_n, test_size =0.3)
model = SVC()


# In[86]:


model.fit(x_train, y_train) 


# In[87]:


model.score(x_test, y_test) # %05


# In[89]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)


# In[90]:


knn.score(x_test, y_test)

