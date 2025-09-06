# 🦶 Advanced Shoe Recommendation System with RAG Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44.2-yellow)](https://huggingface.co/transformers/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-green)](https://faiss.ai/)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive-orange)](https://gradio.app/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-red)](https://ai.google.dev/)

> A state-of-the-art Retrieval-Augmented Generation (RAG) system that provides intelligent, personalized shoe recommendations through natural language understanding and semantic search.

## 🌟 Key Features

### 🧠 **Intelligent Query Understanding**
- **Natural Language Processing**: Advanced NLP pipeline using spaCy for entity recognition
- **Multi-attribute Parsing**: Automatically extracts gender, occasion, brand, size, price range, and rating preferences
- **Semantic Understanding**: Comprehends user intent beyond keyword matching

### 🔍 **Advanced Retrieval System**
- **Semantic Search**: Powered by sentence-transformers for context-aware similarity matching
- **GPU-Accelerated FAISS**: Lightning-fast vector similarity search with GPU optimization
- **Hybrid Filtering**: Combines structured filtering with semantic search for precise results

### 🤖 **Generative AI Integration**
- **RAG Architecture**: Retrieval-Augmented Generation for contextually relevant recommendations
- **Google Gemini Integration**: Leverages Gemini 1.5 Flash for natural language generation
- **Prompt Compression**: Uses LLMlingua for efficient prompt optimization

### 💻 **User Experience**
- **Interactive Web Interface**: Beautiful Gradio-based chat interface
- **Real-time Responses**: Instant recommendations with conversational AI
- **Multi-modal Input**: Supports various query formats and natural language expressions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  NLP Processing  │───▶│   Entity        │
│  "Nike running  │    │  • spaCy NER     │    │  Extraction     │
│   shoes size 9" │    │  • Tokenization  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Response │◄───│  Gemini LLM      │◄───│  Data Filtering │
│   Generation    │    │  Generation      │    │  & Retrieval    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                       │
         │              ┌─────────────────┐              │
         └──────────────│ Prompt          │◄─────────────┘
                        │ Compression     │
                        │ (LLMlingua)     │    ┌─────────────────┐
                        └─────────────────┘    │  FAISS Vector   │
                                              │  Search         │
                        ┌─────────────────┐    │  (GPU Accel.)   │
                        │  Embeddings     │───▶│                 │
                        │  Generation     │    └─────────────────┘
                        │  (MiniLM-L12)   │
                        └─────────────────┘
```

## 📊 Dataset Overview

The system utilizes a comprehensive shoe dataset with **46,879 products** containing:

| Feature | Description | Example Values |
|---------|-------------|----------------|
| **Gender** | Target demographic | male, female |
| **Occasion** | Use case | formal, casual, sports, wedding, ethnic |
| **Brand** | Manufacturer | Nike, Adidas, Puma, etc. |
| **Size** | Available sizes | [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] |
| **Rating** | Customer rating | 3.8, 4.2, 4.5 (out of 5) |
| **Reviews** | Number of reviews | 12953, 40907, 6334 |
| **Description** | Product details | "lace up for men", "running shoes" |
| **Price** | Current price | 783.0, 499.0, 537.0 |
| **Discount** | Offer percentage | 60%, 50%, 78% |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Google AI API key for Gemini

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sureshkannan0919/Shoe_recommendation_system-RAG-.git
   cd Shoe_recommendation_system-RAG-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set up Google AI API**
   ```python
   # Replace "GEMINI-API-KEY" in the notebook with your actual API key
   genai.configure(api_key="your-gemini-api-key-here")
   ```

### Running the System

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook shoe_recommendation_system(RAG).ipynb
   ```

2. **Execute all cells** to initialize the system

3. **Access the web interface** at the provided local URL (typically `http://127.0.0.1:7860`)

## 🎯 Usage Examples

### Natural Language Queries

The system understands various query formats:

```
🔍 "Show me Nike running shoes for men in size 9"
🔍 "I need formal black shoes under $500"
🔍 "Casual shoes for women with good ratings"
🔍 "Wedding shoes with at least 4-star rating"
🔍 "Sports shoes with 50% or more discount"
```

### Query Capabilities

| Query Type | Example | Extracted Entities |
|------------|---------|-------------------|
| **Brand + Gender** | "Adidas shoes for men" | brand: adidas, gender: male |
| **Price Range** | "shoes under 1000" | cprice: 1000 |
| **Occasion + Size** | "formal shoes size 8" | occasion: formal, size: 8.0 |
| **Rating Filter** | "shoes with rating above 4" | rating_threshold: 4.0 |
| **Discount** | "shoes with 60% off" | offer: 60.0 |

## 🔧 Technical Implementation

### 1. **Embedding Generation**
- **Model**: `sentence-transformers/all-MiniLM-l12-v2`
- **Dimension**: 384-dimensional vectors
- **Batch Processing**: Optimized for large-scale embedding generation
- **GPU Acceleration**: CUDA support for faster processing

### 2. **Similarity Search**
- **Index Type**: FAISS IndexFlatL2
- **Distance Metric**: L2 (Euclidean) distance
- **Search Efficiency**: O(n) complexity with GPU optimization
- **Top-K Retrieval**: Configurable number of similar items

### 3. **Natural Language Generation**
- **Model**: Google Gemini 1.5 Flash
- **Context Window**: Optimized with prompt compression
- **Temperature**: 1.0 for balanced creativity and accuracy
- **Max Tokens**: 8192 for comprehensive responses

### 4. **Performance Optimizations**
- **Prompt Compression**: LLMlingua for efficient context utilization
- **Batch Processing**: Vectorized operations for speed
- **GPU Memory Management**: Efficient CUDA resource utilization
- **Caching**: Example caching for improved response times

## 🧪 System Workflow

1. **Query Processing**
   ```python
   query = "Nike running shoes for men size 9"
   tokens = preprocess_query(query)
   entities = parse_query(query, df)
   ```

2. **Data Filtering**
   ```python
   filtered_data = filter_shoes(df, entities)
   ```

3. **Semantic Search**
   ```python
   query_embedding = generate_text_embedding(query)
   distances, indices = similarity_search(embeddings_df, query)
   ```

4. **Result Ranking**
   ```python
   results = filtered_data.iloc[indices[0]]
   results['distance'] = distances.flatten()
   ```

5. **Response Generation**
   ```python
   compressed_prompt = compress_query_prompt(results)
   response = handle_user_query(query, compressed_prompt)
   ```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 46,879 products |
| **Embedding Dimension** | 384 |
| **Search Latency** | < 100ms (GPU) |
| **Embedding Generation** | ~47.75s (batch) |
| **Memory Usage** | ~2GB (GPU) |
| **Accuracy** | High semantic relevance |

## 🛠️ Configuration Options

### Model Configuration
```python
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}
```

### Search Parameters
```python
k = 10  # Number of recommendations
batch_size = 32  # Embedding batch size
```

### GPU Settings
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## 📚 Dependencies

### Core Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers
- `faiss` - Efficient similarity search

### NLP Libraries
- `spacy` - Advanced NLP processing
- `nltk` - Natural language toolkit
- `sentence-transformers` - Semantic embeddings

### AI/ML Libraries
- `google-generativeai` - Google Gemini integration
- `llmlingua` - Prompt compression
- `gradio` - Web interface

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Contribution Areas
- 🔧 Performance optimizations
- 📊 Additional datasets
- 🎨 UI improvements
- 🧠 Algorithm enhancements
- 📖 Documentation updates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **[Hugging Face](https://huggingface.co/)** - For the excellent Transformers library and pre-trained models
- **[Google AI](https://ai.google.dev/)** - For the powerful Gemini generative AI model
- **[FAISS](https://faiss.ai/)** - For efficient similarity search and clustering
- **[Gradio](https://gradio.app/)** - For the intuitive web interface framework
- **[spaCy](https://spacy.io/)** - For advanced natural language processing capabilities

## 📞 Contact

**Suresh Kannan** - [GitHub Profile](https://github.com/Sureshkannan0919)

Project Link: [https://github.com/Sureshkannan0919/Shoe_recommendation_system-RAG-](https://github.com/Sureshkannan0919/Shoe_recommendation_system-RAG-)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ by [Suresh Kannan](https://github.com/Sureshkannan0919)

</div>