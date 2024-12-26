# RAG Model for Shoes Recommendation System

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) model tailored for recommending shoes to users. The system leverages natural language processing (NLP), machine learning, and generative AI to provide personalized and context-aware recommendations.

## Features
- **Natural Language Understanding**: Leverages NLP to analyze user inputs.
- **Semantic Search**: Uses FAISS for efficient vector similarity searches.
- **Generative AI**: Employs a RAG approach to generate personalized shoe recommendations.
- **Interactive Interface**: Integrated with Gradio for a user-friendly web-based interface.



## Getting Started

### Prerequisites
Make sure you have Python 3.8 or higher installed along with the following libraries:
- `pandas`
- `numpy`
- `spacy`
- `nltk`
- `transformers`
- `torch`
- `faiss`
- `gradio`
- `google-generativeai`
- `llmlingua`

To install the required libraries, run:
```bash
pip install pandas numpy spacy nltk transformers torch faiss-gpu gradio google-generativeai llmlingua
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-shoe-recommendation.git
   cd RAG-shoe-recommendation
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
1. Ensure your dataset contains relevant features for shoe recommendations (e.g., user preferences, shoe attributes).
2. Preprocess the data using the provided `data_preprocessing.py` script.

 Input**: The system takes user preferences as input through the Gradio interface.
2. **Semantic Understanding**: Processes the input using NLP and tokenization techniques.
3. **Information Retrieval**: FAISS retrieves relevant embeddings from the knowledge base.
4. **Recommendation Generation**: A generative AI model (via `google.generativeai` and `llmlingua`) generates personalized shoe recommendations.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code adheres to the project's style and standards.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- [Hugging Face](https://huggingface.co/) for the Transformers library.
- [Google Generative AI](https://ai.google/) for powering the recommendation engine.
- [FAISS](https://faiss.ai/) for efficient similarity search.
- [Gradio](https://gradio.app/) for the interactive UI.
# Shoe_recommendation_system-RAG-
