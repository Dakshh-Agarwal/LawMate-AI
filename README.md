# ⚖️ LawMate AI - Intelligent Legal Assistant

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-18.2.0-blue?logo=react)
![Vite](https://img.shields.io/badge/Vite-5.0.8-646CFF?logo=vite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?logo=pytorch&logoColor=white)

LawMate AI is a cutting-edge, AI-powered legal consultation assistant. Built with a modern React + Vite frontend and a high-performance FastAPI + PyTorch backend, it leverages Retrieval-Augmented Generation (RAG) and advanced Language Models to provide accurate, context-aware legal insights.

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
