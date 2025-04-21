#  Plot-Based Movie Recommendation System Using NLP

This project builds a **movie recommendation system** that suggests similar movies based on **plot summaries**, using **Sentence-BERT (SBERT)** embeddings to capture semantic similarity.

Traditional systems rely on user ratings or metadata like genres or cast. This project enhances recommendation quality by analyzing the **narrative content** of movie plots using state-of-the-art NLP techniques.

## Note
- If the ipynb file is not properly visible in the gitHub, click on the provided link to [view the code](https://colab.research.google.com/drive/1KfUZMHvud7THHsON3lgw_zR8SKkBfPTj?usp=sharing)
---

## Features

- NLP-driven content-based recommendations using SBERT
- Cosine similarity to compute semantic closeness
- Scalable nearest neighbor search using FAISS
- Evaluated using Precision@k, Recall@k, and MRR
- Simple text-based search interface (not a web app)

---

##  Project Structure

```bash
.
├── MovieRecommender.ipynb     # Main Jupyter notebook
├── data/
│   ├── tmdb_1000_movies.csv             # Final cleaned dataset with metadata and plot summaries
│   └── embeddings.faiss       # FAISS index for similarity search (if saved)
├── models/
│   └── sbert_model/           # Sentence-BERT model files (auto-downloaded via Transformers)
├── README.md                  # Project documentation
```

## Technologies Used
Python 3.8+

- Sentence-BERT
- Transformers (HuggingFace)
- FAISS (Facebook AI Similarity Search)
- Pandas, Scikit-learn, NumPy

## How to Run
- Ensure movies.csv is in the data/ folder. It should include columns like:
  - title, overview, genres, release_date, etc.

```bash
jupyter notebook MovieRecommender.ipynb
````
### Follow the notebook cells in order:
- Load and clean data
- Generate Sentence-BERT embeddings
- Build FAISS index
- Input a plot or movie title to get recommendations

```python
query = "A thief who enters people's dreams to steal ideas"
recommend_movies(query, top_k=5)
```

```python
Top 5 Recommendations:
1. Inception
2. The Matrix
3. Interstellar
...
```
##  Evaluation
The model is evaluated using:
- Precision@k
- Recall@k
- MRR (Mean Reciprocal Rank)
Evaluation is based on similarity ranking vs known relevant matches from test queries.

```python
# Example test cases (movie title → relevant movie titles)
evaluation_set = [
    {
        "query": "The batman",
        "relevant_titles": ["The dark Knight", "The Batman", "The dark knight rises"]
    },
    {
        "query": "A thief who enters people's dreams to steal ideas",
        "relevant_titles": ["Inception", "The prestige", "Memento"]
    },
    {
        "query":"An elderly man reads to a woman with dementia the story of two young lovers whose romance is threatened by the difference in their respective social classes",
        "relevant_titles" : ["The notebook","Pride & Prejudice","La La Land"]
    }
]
```
## Future Work
- Fine-tune SBERT on domain-specific movie plots
- Incorporate user interaction data for hybrid recommendations
- Build an interactive web app using Flask or Streamlit

## Acknowledgements
- The Movie Database (TMDb)
- Sentence-BERT

## References
1. Getting started. The Movie Database (TMDB). (n.d.). https://www.themoviedb.org/documentation/api  

2. Johnson, J., Douze, M., & Jégou, H. (2017, February 28). Billion-scale similarity search with gpus. arXiv.org. https://arxiv.org/abs/1702.08734  

3. Reimers, N., & Gurevych, I. (2019, August 27). Sentence-bert: Sentence embeddings using Siamese Bert-Networks. arXiv.org. https://arxiv.org/abs/1908.10084

##  Author
Dennis Mathew Jose
MS in Data Analytics Engineering, Northeastern University
