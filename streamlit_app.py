import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from PIL import Image
import base64

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()

    y = [ps.stem(i) for i in text]

    return " ".join(y)


# Load the vectorizer and model
try:
    with open('vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Load and display the image



# Convert jpg image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


background_image = get_base64_image("qwe.jpg")

# Custom CSS with background image and text box color changes
st.markdown(
    f"""
    <style>
    .main {{
        background-image: url('data:image/jpeg;base64,{background_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .title {{
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
    }}
    .stTextArea > div > textarea {{
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        color: black;  /* Ensure text inside the box is visible */
    }}
    .stButton > button {{
        background-color: #667BC6;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }}
    .header {{
        text-align: center;
        font-size: 2em;
        margin-top: 20px;
    }}
    .css-1v3fvcr.e16nr0p30 {{
        background-color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply the background to the main content
st.markdown('<div class="main">', unsafe_allow_html=True)

# Title and input
st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From :-", ["Via Email", "Via SMS", "Other"])

if st.checkbox("Check me"):
    st.write("")

if st.button('Click to Predict'):
    if input_sms:
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header('Not Spam')
    else:
        st.error("Please enter a message to classify.")

# Close the main content div
st.markdown('</div>', unsafe_allow_html=True)
