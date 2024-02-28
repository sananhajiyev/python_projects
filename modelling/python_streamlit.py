# Dataset Description:<br>
# 
# age: continuous.<br>
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.<br>
# fnlwgt: continuous.<br>
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.<br>
# education-num: continuous.<br>
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.<br>
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.<br>
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.<br>
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.<br>
# sex: Female, Male.<br>
# capital-gain: continuous.<br>
# capital-loss: continuous.<br>
# hours-per-week: continuous.<br>
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying- US(Guam-USVI-etc.), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy,Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El- Salvador, Trinadad & Tobago, Peru, Hong, Holland-Netherlands.<br>
# 
# ## Task: You asked to create project with using sklearn Pipeline and ColumnTransformer.


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import copy


data = pd.read_csv("income_evaluation.csv")


df = copy.deepcopy(data)
df.columns = df.columns.str.strip()


X = df.drop("income", axis=1)
y = df["income"]


categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression()

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', classifier)])
pipeline.fit(X_train, y_train)


st.title('Income Prediction App')

st.sidebar.header('Model Parameters')
st.sidebar.write('You can customize the input features.')

age = st.sidebar.slider('Age', min_value=17, max_value=90, value=35)
education_num = st.sidebar.slider('Education Number', min_value=1, max_value=16, value=10)
capital_gain = st.sidebar.slider('Capital Gain', min_value=0, max_value=99999, value=0)
capital_loss = st.sidebar.slider('Capital Loss', min_value=0, max_value=4356, value=0)
hours_per_week = st.sidebar.slider('Hours per Week', min_value=1, max_value=99, value=40)

workclass = st.sidebar.selectbox('Workclass', df['workclass'].unique())
education = st.sidebar.selectbox('Education', df['education'].unique())
marital_status = st.sidebar.selectbox('Marital Status', df['marital-status'].unique())
occupation = st.sidebar.selectbox('Occupation', df['occupation'].unique())
relationship = st.sidebar.selectbox('Relationship', df['relationship'].unique())
race = st.sidebar.selectbox('Race', df['race'].unique())
sex = st.sidebar.selectbox('Sex', df['sex'].unique())
native_country = st.sidebar.selectbox('Native Country', df['native-country'].unique())


user_input = pd.DataFrame({
    'age': [age],
    'education-num': [education_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'native-country': [native_country]
})

prediction = pipeline.predict(user_input)

st.write('## Prediction')
if prediction[0] == '<=50K':
    st.write('The model predicts that the income is **<=50K**.')
else:
    st.write('The model predicts that the income is **>50K**.')


st.sidebar.write('## Model Performance')


y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
st.sidebar.write(f"Accuracy: {round(accuracy, 3) * 100}%")


st.sidebar.write("Classification Report:")
st.sidebar.code(classification_rep)