import streamlit as st
import pickle
import os
import re
import spacy
from spacy.tokens import Span
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import string
from spacy.lang.en.stop_words import STOP_WORDS
from PIL import Image
import matplotlib.pyplot as plt
import spacy_streamlit
import en_core_web_sm

# Provide the path to the downloaded model directory
model_path = r'G:\DATA SCIENCE-25\Github\entity_senti_git\en_core_web_sm-1.2.0'

# Load the model
nlp = spacy.load(model_path)


# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load RandomForestClassifier model
rf = RandomForestClassifier()

# Define custom entities
custom_entities = ['Borderlands','Overwatch','Xbox(Xseries)','TomClancysGhostRecon','Dota2', 'CS-GO',
                   'AssassinsCreed','ApexLegends','LeagueOfLegends','Fortnite','Hearthstone','Battlefield',
                   'PlayerUnknownsBattlegrounds','PUBG','CallOfDuty','TomClancysRainbowSix','GrandTheftAuto(GTA)',
                   'Cyberpunk2077']

# Add custom entity ruler to the pipeline
ruler = nlp.add_pipe("entity_ruler")

# Define patterns for custom entities
patterns = [
    {"label": "PERSON", "pattern": [{"LOWER": "borderlands"}]},
    {"label": "PERSON", "pattern": [{"LOWER": "playerunknownsbattlegrounds"}]},
    {"label": "PERSON", "pattern": [{"LOWER": "tomclancysrainbowsix"}]},
    {"label": "PERSON", "pattern": [{"LOWER": "hearthstone"}]},
    {"label": "GAME", "pattern": [{"LOWER": "pubg"}]},
    {"label": "GAME", "pattern": [{"LOWER": "xbox(xseries)"}]},
    {"label": "GAME", "pattern": [{"LOWER": "overwatch"}]},
    {"label": "GAME", "pattern": [{"LOWER": "leagueoflegends"}]},
    {"label": "GAME", "pattern": [{"LOWER": "apexlegends"}]},
    {"label": "GAME", "pattern": [{"LOWER": "callofduty"}]},
    {"label": "GAME", "pattern": [{"LOWER": "battlefield"}]},
    {"label": "ORG", "pattern": [{"LOWER": "dota2"}]},
    {"label": "ORG", "pattern": [{"LOWER": "cs-go"}]},
    {"label": "ORG", "pattern": [{"LOWER": "assassinscreed"}]},
    {"label": "ORG", "pattern": [{"LOWER": "fortnite"}]},
    {"label": "ORG", "pattern": [{"LOWER": "grandtheftauto(gta)"}]},
    {"label": "ORG", "pattern": [{"LOWER": "cyberpunk2077"}]}
]


# Add patterns to the ruler
ruler.add_patterns(patterns)

# Function to perform entity recognition on a text
def entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Define function to clean text
def clean_text(text):
    # Define the pattern to remove unwanted substrings and symbols
    pattern = r"[^a-zA-Z]+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    
    # Remove unwanted patterns from the text using the defined pattern
    cleaned_text = re.sub(pattern, ' ', text)
    
    # Convert text to lowercase
    cleaned_text = cleaned_text.lower()
    
    return cleaned_text

# Define function for tokenization

# Access the default stop words from the loaded model
stop_words = nlp.Defaults.stop_words

# Define punctuation
punctuations = string.punctuation

def spacy_token(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() for word in doc]

    # Removing stop words and punctuation
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    
    # Joining tokens back into a sentence
    processed_sentence = " ".join(mytokens)

    # return preprocessed sentence
    return processed_sentence

def load_entity_functions():
    return entity_recognition

def load_text_processing_functions():
    return clean_text, spacy_token

# Load the model
rf_model = r"G:\DATA SCIENCE-25\Github\entity_senti_git\model\rf.pkl"

# Load the model using rb
with open(rf_model, 'rb') as file:
    rf_loaded = pickle.load(file)

# Main function

def main():
    #st.title("Entity-Level Sentiment Analysis")

    # Path to your image file
    image_path = r"G:\DATA SCIENCE-25\Github\entity_senti_git\static\pic_for_display_01.jpg"
    # Display the title
    #st.title("Your Image Title")
    # Display the image
    st.image(image_path, use_column_width=True)

    text_input = st.text_area("Enter text for analysis", "")

    if st.button("Submit"):
        if text_input:
            # Load model and text processing functions
            model_loaded = rf_loaded 
            clean_text, spacy_token = load_text_processing_functions()
            cleaned_text = clean_text(text_input)
            spacy_text = spacy_token(cleaned_text)
            entity = entity_recognition(cleaned_text)
            # Get embeddings
            embeddings = model.encode([spacy_text])
        
            # Make prediction
            prediction = model_loaded.predict(embeddings)
        
            # Format entity recognition and sentiment prediction
            entity_str = ", ".join([f"{ent[0]} ({ent[1]})" for ent in entity])
            sentiment_label = "Positive" if prediction[0] == 1 else "Negative"
            # Display image based on sentiment prediction
            if prediction[0] == 1:
                image_path = r"G:\DATA SCIENCE-25\Github\entity_senti_git\static\positive.jpg"  # Replace with the path to your positive image
            else:
                image_path = r"G:\DATA SCIENCE-25\Github\entity_senti_git\static\negative.jpg"  # Replace with the path to your negative image
            
            image = Image.open(image_path)
            st.image(image, width=100)

        
            # Customize appearance using Markdown
            result_str = f"**Entity Recognition:** <span style='color:blue'>{entity_str}</span><br>"
            result_str += f"**Predicted Sentiment:** <span style='font-size:20px; color:{'green' if sentiment_label == 'Positive' else 'red'}'>{sentiment_label}</span>"
        
            # Display result using Markdown
            st.markdown(result_str, unsafe_allow_html=True)
        
        else:
            st.write("Please enter some comments before classifying.")    

# Larger gap using multiple <br> tags
    st.markdown("", unsafe_allow_html=True)

    # Instructions for SMS Spam Classifier
    st.write("Welcome to the Entity-Level Sentiment Analysis!")
    st.write("To determine whether the text is Positive or Negative with know it's entity, enter some text into the box.")
    st.write("Click the 'Classify' button, and the result (Entity + Positive or Negative) will be displayed above.")

# Run the app
if __name__ == "__main__":
    main()
