# Token Counter and Embedding Visualizer

This project demonstrates some of the core concepts behind Large Language Models (LLMs), including tokenization, sentence embeddings, semantic search, and embedding visualization.

It is built using Python and Streamlit as a learning project to better understand how modern AI systems process and retrieve information.

## Features

* Count characters, words, and tokens for each sentence
* Display token IDs generated using OpenAI's tokenizer
* Convert sentences into embedding vectors
* Perform semantic search using cosine similarity
* Visualize sentence embeddings in two dimensions using PCA
* Interactive user interface built with Streamlit

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* tiktoken
* Sentence Transformers
* Scikit-learn

## Project Workflow

1. Enter multiple sentences.
2. Convert each sentence into tokens.
3. Count the number of tokens.
4. Generate embeddings for each sentence.
5. Convert the user's search query into an embedding.
6. Compare the query with all sentence embeddings using cosine similarity.
7. Return the most similar sentences.
8. Visualize the embeddings using PCA.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/token_embedding_visualizer.git
```

Move into the project directory:

```bash
cd token_embedding_visualizer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Example Queries

* Tell me about animals
* Programming languages
* Artificial intelligence
* Deep learning
* Food
* Space
* Vehicles



## Author

Vaishnavi Mukala
