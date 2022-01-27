import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def cleaningPartOne():
    
    #Get data down to manageable size, keep predictors that are desired.
    main = pd.read_csv('data/title.principals.tsv', sep='\t')
    main = main[(main['category'] == 'actor') | (main['category'] == 'actress') | (main['category'] == 'director')]
    main = main[['tconst','nconst']]
    main.to_csv('clean_data/main.csv', index = False)
    
    #Only take movies that were made in 1990 and up. 
    #Also had to convert some data to from strings to integers
    titles = pd.read_csv('data/title.basics.tsv', sep='\t')
    titles = titles[(titles['titleType'] == 'movie')]
    titles = titles[(titles['startYear'].astype(str).str.isdigit())]
    titles['startYear'] = titles['startYear'].astype(int)
    titles = titles[titles['startYear'] > 1989]
    titles = titles[['tconst','originalTitle','startYear']]
    titles.to_csv('clean_data/titles.csv', index = False)
    
    #Ratings are the dependent variable, filtered to 
    ratings = pd.read_csv('data/title.ratings.tsv', sep='\t')
    ratings = ratings[ratings['numVotes'] > 999]
    ratings.to_csv('clean_data/ratings.csv', index = False)
    
    #Take only what's needed
    names = pd.read_csv('data/name.basics.tsv', sep='\t')
    names = names[['nconst', 'primaryName']]
    
    #Merge all movies that have a rating, title, and data on their actors/actresses/director
    movies = ratings.merge(titles, how='inner', on='tconst')
    movies = movies.merge(main, how='inner', on='tconst')
    movies = movies.merge(names, how='inner', on='nconst').sort_values(by=['tconst']).reset_index(drop=True)
    
    #Roughly 80% of movies have only 1 person as a feature. 
    #We will take only movies with people who are in at least 3 movies.
    namesOver2 = pd.DataFrame(movies['nconst'].value_counts()[movies['nconst'].value_counts() > 2]).reset_index().drop('nconst', axis=1)
    namesOver2.columns = ['nconst']
    
    #Merge movies with the list of people in at least 3 movies
    output = movies.merge(namesOver2, how = 'inner', on='nconst').sort_values(by=['tconst']).reset_index(drop=True)
    output.to_csv('clean_data/masterdf.csv', index=False)

def cleaningPartTwo():
    
    #Goal is to create a data frame of dummy variables, where each on corresponds to an actor.
    df = pd.read_csv('clean_data/masterdf.csv')
    
    #Get dummy variables in place of where names were.
    dums = pd.get_dummies(df['nconst'])
    
    #Sum columns where movies are repeated so each movie is it's own row.
    data = pd.concat([df['tconst'],dums],axis=1).groupby('tconst').sum().reset_index()
    
    #Get each movie in it's own row without people attached
    dfNoDuplicates = df.drop(['nconst','primaryName'], axis=1).drop_duplicates()
    
    #Merge the two
    output = dfNoDuplicates.merge(data, how='inner', on='tconst')
    output.to_csv('clean_data/modelData.csv', index=False)

def createModel():
    #Create the model
    df = pd.read_csv('clean_data/modelData.csv')
    X = df.iloc[:,5:]
    y = df['averageRating']
    model = Ridge(alpha=0.01)
    model.fit(X, y)
    test_predictions = model.predict(X)
    
    #Save the model
    filename = "model.pkl"  
    with open(filename, 'wb') as file:  
        pickle.dump(model, file)
    
    #Create a file that has the predicted ratings with movie information
    predictions = pd.concat([df.iloc[:,:5],pd.DataFrame(data=test_predictions, columns=['predictRating'])], axis=1)
    predictions['residuals'] = predictions['averageRating'] - predictions['predictRating']
    predictions.to_csv('clean_data/predictedData.csv', index=False)
    
    #Create a file that has individual's average predicted score, number of movies, and coefficient
    data = pd.read_csv('clean_data/masterdf.csv')
    personScore = data.groupby('nconst').mean().reset_index().merge(data[['nconst','primaryName']].drop_duplicates(), how='inner', on='nconst')
    personScore = personScore.drop(['numVotes','startYear'], axis=1)
    personScore['coefficients'] = pd.DataFrame(model.coef_)
    counts = pd.DataFrame(data['nconst'].value_counts()).reset_index()
    counts.columns = ['nconst','count']
    personScore = personScore.merge(counts, how='inner', on='nconst')
    personScore['predictRating'] = data.merge(predictions[['tconst','predictRating']], how='inner', on='tconst').groupby('nconst').mean().reset_index()['predictRating']
    personScore.to_csv('clean_data/personScore.csv', index=False)

def analysis():
    df = pd.read_csv('clean_data/masterdf.csv')
    predData = pd.read_csv('clean_data/predictedData.csv')
    personScore = pd.read_csv('clean_data/personScore.csv')
    pSnormal = personScore[abs(personScore['coefficients']) < 1000000000000]
    with open('model.pkl', 'rb') as file:  
        model = pickle.load(file)
    
    n = predData.shape[0]
    p = personScore.shape[0]
    r2 = r2_score(predData['averageRating'], predData['predictRating'])
    r2a = 1 - ((1-r2)*(n-1))/(n-p-1)
    print('R^2 adjusted', r2a)
    print('Mean Square Error', mean_squared_error(predData['averageRating'], predData['predictRating']))
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.scatterplot(x=df['numVotes'],y=df['averageRating'], alpha=0.1)
    plt.xlabel('Number of Votes')
    plt.ylabel('Average Rating of Movie')
    plt.savefig('graphs/graph_01')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.histplot(df['averageRating'], binwidth=0.1)
    plt.xlabel('Average Rating of Movie')
    plt.ylabel('Frequency')
    plt.savefig('graphs/graph_02')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.histplot(df['nconst'].value_counts(), bins=20)
    plt.xlabel('Number of Appearances in Productions')
    plt.ylabel('Frequency')
    plt.savefig('graphs/graph_03')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(range(1,10), range(1,10), 'k')
    sns.scatterplot(x=predData['averageRating'],y=predData['predictRating'])
    plt.xlabel('Average Rating of Movie')
    plt.ylabel('Predicted Average Rating of Movie')
    plt.savefig('graphs/graph_04')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.histplot(predData['predictRating'], binwidth=0.1)
    plt.xlabel('Predicted Rating of Movie')
    plt.ylabel('Frequency')
    plt.savefig('graphs/graph_05')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.histplot(predData['averageRating'], binwidth=0.1, color='r')
    sns.histplot(predData['predictRating'], binwidth=0.1)
    plt.xlabel('Predicted Rating of Movie (blue) vs Actual Average Rating of Movie (red)')
    plt.ylabel('Frequency')
    plt.savefig('graphs/graph_06')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.scatterplot(x=predData.index,y=predData['residuals'])
    plt.xlabel('Residual Value')
    plt.ylabel('Residual Index')
    plt.savefig('graphs/graph_07')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.histplot(predData['residuals'])
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig('graphs/graph_08')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.scatterplot(x=pSnormal['averageRating'],y=pSnormal['coefficients'])
    plt.xlabel('Average Rating of Person')
    plt.ylabel('Coefficients')
    plt.savefig('graphs/graph_09')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.scatterplot(x=pSnormal['count'],y=pSnormal['coefficients'])
    plt.xlabel('Number of Productions with Person')
    plt.ylabel('Coefficients')
    plt.savefig('graphs/graph_10')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    sns.scatterplot(x=pSnormal['count'],y=pSnormal['averageRating'])
    plt.xlabel('Number of Productions with Person')
    plt.ylabel('Average Rating')
    plt.savefig('graphs/graph_11')
    plt.show()
    
    plt.figure(figsize=(12,8), dpi=100)
    plt.plot(range(1,10), range(1,10), 'k')
    sns.scatterplot(x=pSnormal['averageRating'],y=pSnormal['predictRating'])
    plt.xlabel('Average Rating of person')
    plt.ylabel('Predicted Average Rating of Person')
    plt.savefig('graphs/graph_12')
    plt.show()
    
    sns.pairplot(pSnormal[['averageRating','coefficients','count','predictRating']])
    plt.savefig('graphs/graph_13')
    plt.show()

cleaningPartOne()

cleaningPartTwo()

createModel()

analysis()

