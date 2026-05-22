# ⚖️ LawMate AI - Intelligent Legal Assistant

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-blue?logo=react)
![Vite](https://img.shields.io/badge/Vite-5.0.8-646CFF?logo=vite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)

LawMate AI is a cutting-edge, AI-powered legal consultation assistant. Built with a modern React + Vite frontend and a high-performance FastAPI + PyTorch backend, it leverages Retrieval-Augmented Generation (RAG) and advanced Language Models to provide accurate, context-aware legal insights.

## 🚨 The Problem (Pain Points)

Navigating the legal landscape can be intimidating and expensive for the average individual. Common pain points include:
- **High Costs**: Consulting with a legal professional typically involves steep hourly rates, making basic legal advice inaccessible to many.
- **Complexity**: Legal jargon and massive document troves are difficult for non-experts to interpret and parse effectively.
- **Time-Consuming**: Searching for relevant laws, past precedents, and rights takes hours or days of manual research.
- **Lack of Immediate Help**: Wait times to simply get preliminary guidance can delay critical decision-making.

## 💡 The Solution

**LawMate AI** democratizes access to legal information by providing an intelligent, 24/7 conversational assistant. 
- It instantly searches through vast databases of legal documents to retrieve highly relevant context.
- It translates complex legalese into plain, understandable language.
- It offers immediate preliminary consultations, potentially saving users significant time and money before they ever need to hire a human attorney.

## ✨ Features

- **Conversational Interface**: Real-time chat interface for natural legal consultations.
- **AI-Powered Retrieval**: Uses FAISS and sentence-transformers for fast, similarity-based legal document retrieval.
- **Context-Aware Responses**: Leverages state-of-the-art LLMs (via PyTorch & Transformers) to answer legal queries accurately.
- **Responsive Design**: Fully responsive UI built with Tailwind CSS, providing a seamless experience across all devices.
- **Detailed Legal Reports**: Generates comprehensive partial and final reports based on user queries and legal data.

## 🏗️ Architecture

The application is split into two main components:
1. **Frontend**: A React application bootstrapped with Vite, utilizing Zustand for state management and Tailwind CSS for styling.
2. **Backend**: A robust FastAPI Python server handling vector searches (FAISS) and machine learning model inference.

*(See [ARCHITECTURE.md](./ARCHITECTURE.md) and [DEPLOYMENT_ARCHITECTURE.md](./DEPLOYMENT_ARCHITECTURE.md) for more details).*

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v16+ recommended)
- **Python** (v3.9+ recommended)
- **Git**

### Installation

Clone the repository:
```bash
git clone https://github.com/Dakshh-Agarwal/LawMate-AI.git
cd LawMate-AI
```

### Running the Application (Windows)

We provide convenient PowerShell scripts to quickly start both the frontend and the backend.

**1. Start the Backend Server**
Open a new PowerShell terminal and run:
```powershell
.\backend\start_backend.ps1
```
*This script will automatically create a virtual environment, install the required dependencies from `requirements.txt`, and start the FastAPI server on `http://localhost:8000`.*

**2. Start the Frontend Development Server**
Open another PowerShell terminal and run:
```powershell
.\start_frontend.ps1
```
*This script will install Node modules if they are missing and start the Vite development server on `http://localhost:5173`.*

## 📖 Usage

1. **Start a Consultation**: Open your browser to `http://localhost:5173` and start typing your legal query in the chat interface.
2. **Contextual Retrieval**: LawMate AI will automatically retrieve relevant legal documents in the background based on your query.
3. **Receive Insights**: Read the AI-generated responses tailored to your specific situation, complete with citations or references where applicable.
4. **Generate Reports**: Ask the assistant to compile a detailed report (partial or final) summarizing your consultation, which you can save for future reference or share with a human attorney.

## 🛠️ Technology Stack

**Frontend:**
- [React](https://reactjs.org/) (v18)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Zustand](https://github.com/pmndrs/zustand)
- [Axios](https://axios-http.com/)
- [React Hot Toast](https://react-hot-toast.com/)

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/) & [Transformers](https://huggingface.co/docs/transformers/index)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- [Uvicorn](https://www.uvicorn.org/)

## 🔮 Future Limitations & Roadmap

While LawMate AI is a powerful tool, it currently has some limitations and areas for future improvement:
- **Not Legal Advice**: LawMate AI provides legal *information*, not certified legal *advice*. It does not replace the counsel of a qualified attorney.
- **Jurisdictional Boundaries**: Currently, the vector database may be limited to specific jurisdictions or subsets of law. Expanding the dataset to cover global/regional laws is a priority.
- **Hallucination Risks**: Like all LLMs, there is a small risk of hallucinated precedents. Continuous model fine-tuning and strict RAG prompt-engineering are ongoing to mitigate this.
- **Upcoming Features**:
  - Multi-language support for international accessibility.
  - User accounts and consultation history saving.
  - Integration with live legal databases and APIs.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/Dakshh-Agarwal/LawMate-AI/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
