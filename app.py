import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
import plotly.express as px

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


st.set_page_config(
    page_title="Token Counter + Embedding Visualizer",
    layout="wide"
)

st.title("Token Counter + Embedding Visualizer")

st.write("""
This project shows how text becomes tokens, how tokens are counted,
how sentences become embeddings, and how semantic search works.
""")


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()
tokenizer = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# User input
# -----------------------------
default_text = """I love dogs.
Dogs are loyal animals.
Puppies are playful and cute.
Cats like to sleep in warm places.
Kittens are small and adorable.
Animals need care and food.

Python is a programming language.
Java is used for software development.
SQL is used to query databases.
Machine learning helps computers learn from data.
Artificial intelligence can solve complex problems.
Neural networks are used in deep learning.

Pizza tastes delicious.
Biryani is a flavorful rice dish.
Pasta is an Italian food.
Burgers are popular fast food.
Ice cream is a sweet dessert.
Coffee gives people energy.

The sun is very hot.
The moon orbits the Earth.
Space contains planets and stars.
Astronauts travel in spacecraft.
Mars is called the red planet.
Galaxies contain billions of stars.

Cars need fuel to run.
Electric vehicles use batteries.
Buses carry many passengers.
Trains travel on railway tracks.
Airplanes fly in the sky.
Ships travel across oceans."""

sentences_input = st.text_area(
    "Enter sentences, one sentence per line:",
    value=default_text,
    height=220
)

query = st.text_input(
    "Search by meaning:",
    value="Tell me about animals"
)

top_k = st.slider("Number of search results", 1, 5, 3)


# -----------------------------
# Process button
# -----------------------------
if st.button("Analyze"):
    sentences = [
        s.strip()
        for s in sentences_input.split("\n")
        if s.strip()
    ]

    if len(sentences) < 2:
        st.warning("Please enter at least 2 sentences.")
        st.stop()

    # -----------------------------
    # Token counting
    # -----------------------------
    token_data = []

    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        decoded_tokens = [
            tokenizer.decode([token_id])
            for token_id in token_ids
        ]

        token_data.append({
            "Sentence": sentence,
            "Character Count": len(sentence),
            "Word Count": len(sentence.split()),
            "Token Count": len(token_ids),
            "Token IDs": token_ids,
            "Decoded Tokens": decoded_tokens
        })

    token_df = pd.DataFrame(token_data)

    st.subheader("1. Token Counter")
    st.dataframe(token_df, use_container_width=True)

    st.info("""
Tokens are the small pieces of text that LLMs read.
A token can be a word, part of a word, a space, or punctuation.
LLMs do not directly read full words like humans.
""")

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = embedding_model.encode(sentences)

    st.subheader("2. Embedding Shape")
    st.write("Each sentence is converted into a vector.")

    st.code(f"Embedding shape: {embeddings.shape}")

    st.info("""
The model converts every sentence into numbers.
These numbers represent meaning.
In this model, each sentence becomes a 384-dimensional vector.
""")

    # -----------------------------
    # Semantic search
    # -----------------------------
    query_embedding = embedding_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    search_df = pd.DataFrame({
        "Sentence": sentences,
        "Similarity Score": similarities
    })

    search_df = search_df.sort_values(
        by="Similarity Score",
        ascending=False
    ).head(top_k)

    st.subheader("3. Semantic Search Results")
    st.dataframe(search_df, use_container_width=True)

    st.info("""
Semantic search compares meaning using cosine similarity.
It can find related sentences even when exact words do not match.
""")

    # -----------------------------
    # PCA visualization
    # -----------------------------
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plot_df = pd.DataFrame({
        "Sentence": sentences,
        "X": reduced_embeddings[:, 0],
        "Y": reduced_embeddings[:, 1],
        "Token Count": token_df["Token Count"]
    })

    st.subheader("4. Embedding Visualization")

    fig = px.scatter(
        plot_df,
        x="X",
        y="Y",
        text="Sentence",
        hover_data=["Sentence", "Token Count"],
        title="Sentence Embeddings Visualized in 2D"
    )

    fig.update_traces(textposition="top center")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
Embeddings originally have 384 dimensions.
PCA reduces them to 2 dimensions so we can visualize them.
Sentences close together usually have similar meaning.
""")

    # -----------------------------
    # Concept summary
    # -----------------------------
    st.subheader("5. What You Built")

    st.write("""
You built a mini version of a RAG retrieval system.

Flow:

Text → Tokens → Token Count → Embeddings → Similarity Search → Visualization

This is the same foundation used in AI search, chatbots, document Q&A,
recommendation systems, and RAG applications.
""")