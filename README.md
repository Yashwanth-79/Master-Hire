# Master Hire - AI-Powered Technical Recruitment Assistant

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red?style=flat&logo=youtube)](https://youtu.be/Co6QF_f_ac8)
[![Live Application](https://img.shields.io/badge/Live-App-blue?style=flat&logo=streamlit)](https://master-hire-talent-scout.streamlit.app/)

## 🚀 Project Overview

**Master Hire** is an intelligent recruitment assistant that enhances the technical hiring process using AI. It features a chatbot-style interface powered by Google's Gemini-1.5-pro LLM and leverages **Retrieval-Augmented Generation (RAG)** to dynamically generate and assess technical interview questions.

## ✨ Key Features

- ✅ **Automated Resume Parsing** - Extracts and structures candidate data.
- 🎯 **Dynamic Technical Question Generation** - Tailored assessments based on skills.
- ⚡ **Real-Time Assessment & Feedback** - Instant evaluation of candidate responses.
- 📊 **Comprehensive Admin Dashboard** - Detailed recruitment analytics.
- 🔒 **Secure Data Handling** - Robust file validation and storage.
- 🗂 **Vector-Based Document Storage** - Efficient search with ChromaDB.

---

## 🛠 Installation & Setup

### 📌 Prerequisites
- Python 3.8+
- Google API Key for Gemini-1.5-pro
- Sufficient storage for ChromaDB vector database

### 📥 Installation Steps

1️⃣ **Clone the repository:**
```bash
git clone https://github.com/YourUsername/Master-Hire.git
cd Master-Hire
```

2️⃣ **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3️⃣ **Install required dependencies:**
```bash
pip install -r requirements.txt
```

4️⃣ **Set up environment variables:**
Create a `.env` file in the project root and add your API key:
```ini
GOOGLE_API_KEY=your_api_key_here
```

5️⃣ **Run the application:**
```bash
streamlit run app.py
```

---

## 📖 Usage Guide

### 🎓 For Candidates
1. Click on **"Candidate Login"** from the welcome screen.
2. Upload your resume (PDF, DOC, DOCX | max 200MB).
3. Provide personal & professional details.
4. Rate proficiency in technical skills.
5. Complete the AI-generated technical assessment.
6. Review results and feedback instantly.

### 🏢 For Administrators
1. Login to the **Admin Dashboard** with credentials.
2. View detailed **candidate analytics**.
3. Export candidate data in **CSV/JSON**.
4. Monitor **assessment metrics & skill distribution**.

---

## 🏗 Technical Details

### 🔹 Core Technologies
- **Frontend**: Streamlit
- **Backend**: Python
- **AI Model**: Google Gemini-1.5-pro
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace's `all-MiniLM-L6-v2`

### 🛠 System Architecture

#### 1️⃣ **Input Processing**
- Resume validation & structured parsing
- User input verification & secure storage

#### 2️⃣ **Question Generation**
- AI-generated questions based on user skills
- Mix of **MCQs** and **text-based** questions
- Automated **scoring & feedback**

#### 3️⃣ **Vector Database & Search**
- Efficient document retrieval via **ChromaDB**
- Scalable **similarity search**
- Persistent **storage for future assessments**

---

## 🔥 Prompt Design

### 📥 Information Gathering
The system collects user data in a staged approach:
```python
stages = {
    "upload_resume": {...},
    "name": {...},
    "email": {...},
    # Additional stages
}
```

### 🧠 Dynamic Question Generation
The system generates relevant questions based on skill level:
```python
prompt = f"""Generate 2 multiple choice questions and 1 text-based question for {tech} at proficiency level {level}/10.

Format for MCQs:
MCQ||||||

Format for text questions:
TEXT||
"""
```

---

## 🔎 Challenges & Solutions

### 1️⃣ **Dynamic Question Generation**
**❌ Challenge:** Ensuring consistent quality & relevance of AI-generated questions.

**✅ Solution:** Implemented structured prompts with validation checks to improve accuracy.

### 2️⃣ **Scalability**
**❌ Challenge:** Managing large resume files and multiple candidates.

**✅ Solution:** ChromaDB for optimized vector storage & enforced file size limits.

### 3️⃣ **Assessment Accuracy**
**❌ Challenge:** Fair evaluation of text-based answers.

**✅ Solution:** Developed robust **scoring system** with **answer normalization & pattern matching**.

### 4️⃣ **Security & Data Protection**
**❌ Challenge:** Preventing unauthorized access to sensitive candidate data.

**✅ Solution:**
- Secure file validation mechanisms
- Session-based authentication
- Rate limiting for login attempts
- Secure credential storage

---

## 🚀 Future Enhancements

🔹 **Automated Resume Parsing using NLP**
🔹 **Soft Skills & Personality Profiling**
🔹 **Multi-Language Support**
🔹 **AI-powered Recruiter Dashboard with Insights**

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch.
3. Make necessary changes and push.
4. Submit a Pull Request (PR).

---

## 📜 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.
