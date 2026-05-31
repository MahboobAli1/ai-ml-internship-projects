# 📰 BERT News Topic Classifier

A production-ready news headline classifier using fine-tuned BERT model. Classifies news into 4 categories: World, Sports, Business, and Science/Technology.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.4% |
| **F1-Score** | 91.8% |
| **Training Samples** | 10,000 |
| **Framework** | Hugging Face Transformers |

## 🎯 Categories

- **World** 🌍 - International news, politics, conflicts
- **Sports** ⚽ - Sports events, athletes, competitions  
- **Business** 💼 - Economics, companies, markets
- **Science/Technology** 🔬 - Research, AI, innovation

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Option 2: Command-Line Inference

```bash
# Run interactive mode
python inference.py

# Or use as a module
python -c "
from inference import NewsClassifier
classifier = NewsClassifier()
result = classifier.predict('Apple announces record earnings')
print(result)
"
```

### Option 3: Google Colab (Training)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `bert_news_classifier_colab.py`
3. Run cells sequentially
4. Download trained model files

## 📁 Project Structure

```
bert-news-classifier/
├── app.py                           # Streamlit web app
├── inference.py                     # Standalone inference script
├── bert_news_classifier_colab.py   # Training notebook for Colab
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
└── bert_news_classifier/            # Trained model (download from Colab)
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer.json
    └── vocab.txt
```

## 💾 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/bert-news-classifier.git
cd bert-news-classifier
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n bert-classifier python=3.8
conda activate bert-classifier
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Trained Model
Get the trained model from Colab training output and place in `bert_news_classifier/` directory.

**Alternative**: The app will use `bert-base-uncased` as fallback if trained model isn't available.

## 🎓 Training (Google Colab)

### Setup
1. Open `bert_news_classifier_colab.py` in Google Colab
2. Run cells in order (Colab will prompt for dependencies)

### Key Features
- ✅ Automatic dataset loading and balancing
- ✅ Stratified sampling (10k samples, balanced across classes)
- ✅ GPU acceleration
- ✅ Validation monitoring
- ✅ Model evaluation with confusion matrix
- ✅ Automatic model saving

### Training Time
- **GPU (Colab T4)**: ~8-12 minutes
- **GPU (Colab V100)**: ~5-7 minutes
- **CPU**: ~45-60 minutes (not recommended)

## 📖 Usage Examples

### Example 1: Web App (Streamlit)
```bash
streamlit run app.py
# Open http://localhost:8501
# Type or paste headlines for instant classification
```

### Example 2: Python Module
```python
from inference import NewsClassifier

# Initialize
classifier = NewsClassifier()

# Single prediction
result = classifier.predict("Stock market crashes amid economic concerns")
print(f"Category: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
headlines = [
    "New AI model beats human performance",
    "Chelsea wins Premier League",
    "Oil prices surge in global markets",
    "UN summit addresses climate crisis"
]

for headline in headlines:
    result = classifier.predict(headline)
    print(f"{headline}")
    print(f"  → {result['predicted_class']} ({result['confidence']:.1%})\n")
```

### Example 3: Direct CLI
```bash
python inference.py
# Type headlines interactively
```

### Example 4: Get Model Info
```python
from inference import NewsClassifier

classifier = NewsClassifier()
info = classifier.get_model_info()
print(info)
```

## 🔧 Customization

### Change Model Architecture
Edit `bert_news_classifier_colab.py`:
```python
CONFIG = {
    "model_name": "bert-large-uncased",  # Change this
    "max_length": 256,                    # Increase if needed
    "num_epochs": 5,                      # More training
    # ... other settings
}
```

### Adjust Streamlit Theme
Edit `app.py` CSS section to customize colors and styling.

### Add More Categories
1. Modify training data in Colab
2. Update `class_names` in config.json
3. Retrain the model

## 🐳 Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t bert-classifier .
docker run -p 8501:8501 bert-classifier
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud
```bash
git push origin main  # Push to GitHub
# Go to https://streamlit.io/cloud
# Connect repository and deploy
```

### Option 2: Heroku
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

### Option 3: AWS
1. Upload to S3
2. Deploy with SageMaker or EC2
3. Use Lambda for serverless

### Option 4: Azure
1. Create Web App
2. Deploy from GitHub
3. Configure App Settings

## ⚙️ Troubleshooting

### Issue: "Model not found"
**Solution**: Download trained model from Colab and place in `bert_news_classifier/` directory

### Issue: Memory error during training
**Solution**: Reduce `batch_size` in CONFIG (try 16 instead of 32)

### Issue: GPU not detected in Colab
**Solution**: Go to Runtime → Change Runtime Type → GPU (T4 or V100)

### Issue: Dependency conflicts
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
# Or create fresh environment
```

### Issue: Streamlit port already in use
**Solution**:
```bash
streamlit run app.py --server.port 8502
```

## 📊 Model Architecture

```
BERT (bert-base-uncased)
├── Token Embeddings (768-dim)
├── 12 Transformer Layers
│   ├── Multi-head Self-Attention (12 heads)
│   └── Feed-Forward Networks
└── Classification Head
    ├── [CLS] Token
    ├── Dropout (0.1)
    └── Linear (768 → 4 classes)
```

## 📈 Performance Metrics

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| World | 0.93 | 0.92 | 0.92 | 500 |
| Sports | 0.95 | 0.94 | 0.94 | 500 |
| Business | 0.91 | 0.92 | 0.91 | 500 |
| Science/Tech | 0.89 | 0.91 | 0.90 | 500 |

## 🔄 Workflow Summary

```
1. TRAINING (Google Colab)
   ↓
   bert_news_classifier_colab.py
   ├── Load AG News dataset (10k samples)
   ├── Fine-tune BERT model
   └── Save weights & tokenizer
   ↓

2. TESTING & SIMULATION
   ↓
   Test headlines locally
   ├── inference.py (CLI)
   └── Validation metrics
   ↓

3. DEPLOYMENT (Streamlit)
   ↓
   app.py
   ├── Web interface
   ├── Real-time classification
   └── Beautiful visualizations
   ↓

4. GITHUB
   ↓
   git push to repository
   └── Ready for production
```

## 📚 Learning Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [Transformers Tutorial](https://huggingface.co/course)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Mahboob Ali**
- 🎓 BS Artificial Intelligence Student
- 🏠 Sukkur, Sindh, Pakistan
- 📧 Contact for questions or collaboration

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- AG News dataset creators
- Streamlit for easy deployment
- BERT authors at Google

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: Your email here

## 🗺️ Roadmap

- [ ] Multi-language support (Arabic, Urdu)
- [ ] Real-time model updates
- [ ] Model compression (DistilBERT)
- [ ] REST API endpoint
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Custom fine-tuning interface

---

**Last Updated**: May 2024
**Maintained By**: Mahboob Ali
**Repository**: https://github.com/yourusername/bert-news-classifier

⭐ If you find this helpful, please star the repository!
