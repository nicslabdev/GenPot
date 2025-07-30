# 🍯 Honeypot with AI

<div align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2lwZTZ3djZzYW54MmY0ZmpwanBvbTRhMnd2YzYxa2FwN25lMnFxYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/I0e4u216Qhww8eRTVq/giphy.gif" width="450" alt="Cute Bee Animation"/>
  
  <br/>
  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
  [![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
  [![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

## 🎯 Project Description

**Honeypot with AI** is an advanced intelligent honeypot system that combines cybersecurity techniques with AI. The project uses fine-tuned language models to detect and respond to security threats autonomously.

## ✨ Key Features

### 🧠 Fine-tuned AI Models
- **Gemma 7B**: Specialized model for threat detection
- **Llama 3 8B**: Intelligent response system
- **Zephyr 7B**: Attacker behavior analysis

### 🛡️ Advanced Honeypot
- **OpenCanary Integration**: Vulnerable service simulation
- **FastAPI Backend**: REST API for management and monitoring
- **Docker Support**: Easy and scalable deployment

### 📊 Monitoring and Analytics
- Real-time energy consumption tracking and carbon footprint
- Model performance metrics
- Detailed activity logs

## 📁 Project Structure

```
🏠 CiberIA_O1_A3/
├── 🍯 honeypot/
│   ├── 🐳 docker-compose.yml
│   ├── 🚀 fastapi/
│   │   ├── fastapi_server.py
│   │   └── 📊 results/
│   ├── 🧠 fine_tuning/
│   │   ├── 📚 datasets/
│   │   ├── 🎯 tuning/
│   │   └── 🛠️ utils/
│   ├── 🤖 models/
│   │   ├── gemma_base/
│   │   ├── gemma_finetuned/
│   │   ├── llama3_base/
│   │   └── zephyr_base/
│   └── 🕯️ opencanary/
└── 📖 README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA compatible GPU (recommended)
- Docker & Docker Compose
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/usuario/CiberIA_O1_A3.git
cd CiberIA_O1_A3

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configurations
```

### 🐳 Docker Deployment

```bash
cd honeypot
docker compose up --build
```

### 🧠 Model Fine-tuning

```bash
cd honeypot/fine_tuning/tuning

# Fine-tune Gemma
python fine_tuning_gemma.py

# Fine-tune Llama 3
python fine_tuning_llama3.py

# Fine-tune Zephyr
python fine_tuning_zephyr.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Hugging Face Models
GEMMA_MODEL_NAME=google/gemma-7b-it
LLAMA_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
ZEPHYR_MODEL_NAME=HuggingFaceH4/zephyr-7b-beta

# Base paths
MODELS_BASE_DIR=/path/to/models
DATASETS_BASE_DIR=/path/to/datasets

...
```

## 📊 Metrics and Monitoring

The system includes advanced tracking of:
- ⚡ Real-time energy consumption
- 🌱 Estimated carbon footprint
- 📈 API response times
- 🎯 Fine-tuned model accuracy

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 👥 Authors

- **Alara** - *Lead Developer* - [@alara](https://github.com/alara)

---

<div align="center">
  <sub>Built with ❤️ for intelligent cybersecurity</sub>
</div>
