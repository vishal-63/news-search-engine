""" 
to start the server run:
        uvicorn api:app --reload 
"""

import os
import heapq
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the word embedding model (GloVe) from gensim
word_embedding_model = api.load("glove-wiki-gigaword-300")


def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize the sentences into words and remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]

    # Calculate the frequency of each word
    freq_dist = FreqDist(words)

    # Calculate the score for each sentence based on the frequency of its words
    sentence_scores = []
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = sum([freq_dist[word] for word in sentence_words])
        sentence_scores.append((sentence, score))

    # Choose the top sentence as the summary
    summary_sentence = heapq.nlargest(
        1, sentence_scores, key=lambda x: x[1])[0][0]

    return summary_sentence


def preprocess(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())

    # Remove stop words from the tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)


def generate_vector_glove(text):
    # Preprocess the text by removing stop words
    preprocessed_text = preprocess(text)

    embeddings = []
    # Iterate over each word in the preprocessed text
    for word in preprocessed_text.split():
        if word in word_embedding_model:
            # If the word is present in the word embedding model, get its vector and append it to the embeddings list
            embeddings.append(word_embedding_model[word])

    if embeddings:
        # If embeddings are found, calculate the mean vector of all embeddings
        return np.mean(embeddings, axis=0)
    else:
        # If no embeddings are found, return a zero vector of the same size as the word embedding model's vector size
        return np.zeros(word_embedding_model.vector_size)


def cosine_similarity(a, b):
    # Calculate the cosine similarity between two vectors a and b
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_glove(query='world', n=6):
    # Read the data from a CSV file
    df = pd.read_csv('data.csv')

    # Generate the word vector for the query
    query_vector = generate_vector_glove(query)

    results = []
    # Iterate over each data row in the dataframe
    for i in range(len(df)):
        data = df.iloc[i]
        vector = generate_vector_glove(data['news'])
        # Calculate the similarity between the query vector and the vector of each data row
        similarity = cosine_similarity(query_vector, vector)
        results.append((data, similarity))

    # Sort the results based on the similarity score in descending order
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Return the top n results
    return [r[0] for r in results[:n]]


# Generate word vectors for different categories
ent_glove = generate_vector_glove('Entertainment Movies songs Party Actor')
politics_glove = generate_vector_glove(
    'Politics election scam Government legislation democracy voting parliament law diplomacy public affairs')
travel_glove = generate_vector_glove(
    'Travel vacation journey airlines hotels resorts hot cold')
business_glove = generate_vector_glove(
    'Business equity firm economy market stocks shares finance investment')
health_glove = generate_vector_glove(
    'Health wellness pharmaceutical symptoms hospital')
style_glove = generate_vector_glove(
    'Style fashion trend designer apparel accessories beauty red carpet')
sport_glove = generate_vector_glove(
    'Sports cricket baseball football leagues championship tournament score')
weather_glove = generate_vector_glove('Weather')

# Create a dictionary of categories and their corresponding word vectors
ListOfCategories = {'Entertainment': ent_glove, 'Politics': politics_glove,
                    'Travel': travel_glove, 'Business': business_glove,
                    'Health': health_glove, 'Style': style_glove, 'Sports': sport_glove,
                    'Weather': weather_glove}


def get_category(text):
    DocBelongsToCat = []
    doc_glove = generate_vector_glove(text)

    # Calculate the similarity between the document vector and the vectors of each category
    for i, j in enumerate(ListOfCategories):
        DocBelongsToCat.append(
            [j, cosine_similarity(ListOfCategories[j], doc_glove)])

    # Sort the categories based on the similarity scores in descending order
    DocBelongsToCat.sort(key=lambda x: x[1], reverse=True)
    print(DocBelongsToCat)
    category = DocBelongsToCat[0]

    return category[0]


@app.get("/search")
async def search(query=None):
    if query is not None:
        # Perform a search based on the query
        articles = search_glove(query)
        results = []
        for article in articles:
            category = article['category']
            link = article['link']
            heading = article['heading']
            news = article['news']
            # Generate the summary for the news article
            summary = summarize_text(news)
            doc_dict = {
                'link': link, 'category': category, 'heading': heading, 'summary': summary}
            results.append(doc_dict)

        return results

    return {None}

if __name__ == '__main__':
    app.run(debug=True, port=8000)
