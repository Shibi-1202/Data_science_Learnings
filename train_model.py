import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


data = pd.read_csv('Movie_Interests.csv')


interest_names = data['Interest'].unique() 
interest_map = {} 
for i in range(len(interest_names)):
    interest_map[interest_names[i]] = i  
data['Interest_encoded'] = data['Interest'].map(interest_map)


X = data[['Age', 'Gender']] 
y = data['Interest_encoded']  


model = DecisionTreeClassifier()
model.fit(X, y)


joblib.dump(model, 'model.pkl')
joblib.dump(interest_map, 'interest_map.pkl')

print('Model and mapping saved successfully.')
