import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
plt.style.use(style='ggplot')


train_dataset = pd.read_csv('train.csv')

test_dataset = pd.read_csv('test.csv')

output = train_dataset[train_dataset.Promoted_or_Not.notnull()]
output



#Convert [Qualification] to [1,0] so that our decision tree can be built
for df in [train_dataset,test_dataset]:
    df['staff_Qualification']=df['Qualification'].map({'MSc, MBA and PhD':1,'First Degree or HND':0})
# Drop the Qualification Column 
train_dataset = train_dataset.drop(['Qualification'], axis =1)
test_dataset = test_dataset.drop(['Qualification'], axis =1)

#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train_dataset['staff_Qualification'] = train_dataset['staff_Qualification'].fillna(0)
test_dataset['staff_Qualification'] = test_dataset['staff_Qualification'].fillna(0)

num_features = train_dataset.select_dtypes(include=[np.number])


# Feature selection
y = num_features.Promoted_or_Not

X = num_features.drop(['Year_of_birth', 'Promoted_or_Not'], axis =1)


#num_features = num_features [num_features['Trainings_Attended', 'Year_of_recruitment ', 'Year_of_birth' ] <100]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 16496)


# Model Fitting
model = SVC(gamma = 'auto')
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = model.score(X_test , y_test)

print(accuracy)

submission = pd.DataFrame({'EmployeeNo': test_dataset['EmployeeNo'], 'Promoted_or_Not':pred})
submission.head(5)

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'submission.csv'

submission.to_csv(filename,index=False)
