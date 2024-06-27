import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv(r"D:\Naresh it\My work\capstone project for resume\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Preprocess the text data
corpus = []
ps = PorterStemmer()
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# Streamlit App
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon=":fork_and_knife:",
    layout="wide"
)

# Define a function for preprocessing user input
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review


st.markdown('<div class="main">', unsafe_allow_html=True)

# Title and description
st.title('Restaurant Review Sentiment Analysis')
st.markdown('### Enter your restaurant review below:')

# Text area for user input
review_text = st.text_area('Input Review', '')

# Prediction and results
if st.button('Predict'):
    if review_text:
        # Preprocess user input
        cleaned_review = preprocess_text(review_text)
        # Vectorize user input
        vectorized_review = tfidf.transform([cleaned_review]).toarray()
        # Predict sentiment
        prediction = classifier.predict(vectorized_review)
        # Display prediction result
        if prediction[0] == 1:
            st.success('This review is positive!')
        else:
            st.error('This review is negative!')
    else:
        st.warning('Please enter a review.')

# Display evaluation metrics conditionally
if st.checkbox('Show Model Evaluation'):
    st.subheader('Model Evaluation')
    st.text(f'Confusion Matrix:\n{confusion_matrix(y_test, classifier.predict(x_test))}')
    st.text(f'Accuracy Score: {accuracy_score(y_test, classifier.predict(x_test)):.4f}')

st.markdown('</div>', unsafe_allow_html=True)
