# T5-Question-Generation-Model

A powerful technical interview question generation system built using the T5 model, designed to automate the creation of relevant technical interview questions from job descriptions. This project leverages a RAG (Retrieval-Augmented Generation) pipeline for context-aware question generation, tailored for technical interviews.

## 🚀 Features

- Technical interview question generation using the T5 model
- RAG pipeline for context-aware question generation
- RESTful API interface with FastAPI
- Docker containerization for easy deployment
- Knowledge base integration with Excel support
- Smart filtering to ensure question quality and relevance

## 📋 Prerequisites

| Tool                  | Version/Requirement         |
|-----------------------|-----------------------------|
| Python                | 3.9 or higher              |
| Docker                | Latest version             |
| Docker Compose        | Latest version             |
| pip                   | Included with Python       |
| Git                   | For cloning the repository |

## 📥 Model Weights

The model weights are not included in this repository due to their size. You need to download them separately:

1. Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1fNGcebiB9zoMl2n7wAiZ5pKlyw01-o0i?usp=sharing)
2. Extract the downloaded zip file
3. Create a directory named `t5_question_gen_model` in the project root
4. Place the extracted model weights in the `t5_question_gen_model` directory

The directory structure should look like this:
```
T5-Question-Generation-model/
├── t5_question_gen_model/    # Model weights directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── generation_config.json
├── api.py                    # FastAPI application
├── rag_pipeline.py          # RAG pipeline implementation
├── knowledge_base.xlsx      # Knowledge base data
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

Note: Make sure to name the directory exactly as `t5_question_gen_model` (all lowercase with underscores) as the code expects this specific directory name.

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/MennaT-T/T5-Question-Generation-model.git
cd T5-Question-Generation-model
```

2. Download and extract the model weights as described in the [Model Weights](#-model-weights) section above

3. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows; use 'source venv/bin/activate' on Unix
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Deployment

1. Download and extract the model weights as described in the [Model Weights](#-model-weights) section above

2. Using Docker Compose (recommended):
```bash
docker-compose up --build
```

3. Or using Docker directly:
```bash
docker build -t t5-question-generator .
docker run -p 8000:8000 t5-question-generator
```

## 🏃‍♂️ Running the Application

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Welcome message
- `POST /generate-questions`: Generate questions from job description
  - Request body:
    ```json
    {
        "description": "Your job description here",
        "num_questions": 3
    }
    ```
  - Response:
    ```json
    {
        "questions": [
            "Question 1?",
            "Question 2?",
            "Question 3?"
        ]
    }
    ```

## 🔧 Configuration

The application uses the following default configurations:
- Port: 8000
- Model: T5
- Knowledge Base: Excel file format
- CORS: Enabled for all origins

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- [Menna Rasslan](https://github.com/MennaT-T)
- [Ahmed Waled](https://github.com/Ahmed-waled)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the T5 model implementation
- FastAPI for the web framework
- Docker for containerization support 