 # Master Hire - AI-Powered Technical Recruitment Assistant
 Demo link = https://youtu.be/Co6QF_f_ac8

 ## Project Overview

Master Hire is an intelligent recruitment assistant that streamlines the technical hiring process using AI. The system provides a chatbot-style interface powered by Google's Gemini-1.5-pro LLM and implements RAG (Retrieval-Augmented Generation) architecture for dynamic technical assessments.

 ### Key Features

 - Automated resume parsing and candidate data collection

 - Dynamic technical question generation based on candidate skills

 - Real-time assessment with immediate feedback

 - Comprehensive admin dashboard for recruitment analytics

 - Secure file handling and data validation

 - Vector-based document storage using ChromaDB

 ## Installation

 ### Prerequisites

 - Python 3.8 or higher

 - Google API key for Gemini-1.5-pro

 - Sufficient storage for ChromaDB vector database

 ### Setup Instructions

1 . Clone the repository:

 ` ` `bash

git clone https://github.com/YourUsername/Master-Hire.git

cd Master-Hire

 ` ` `

2 . Create and activate a virtual environment:

 ` ` `bash

python -m venv venv

source venv/bin/activate # On Windows: venv  Scripts  activate

 ` ` `

3 . Install required packages:

 ` ` `bash

pip install -r requirements.txt

 ` ` `

4 . Create a  `.env ` file in the project root and add your Google API key:

 ` ` `

GOOGLE _API _KEY=your _api _key _here

 ` ` `

5 . Run the application:

 ` ` `bash

streamlit run app.py

 ` ` `

 ## Usage Guide

 ### For Candidates

1 . Click on "Candidate Login" from the welcome screen

2 . Upload your resume (PDF, DOC, or DOCX format, max 200MB)

3 . Fill in your personal and professional details

4 . Rate your proficiency in various technical skills

5 . Complete the generated technical assessment

6 . Review your results and feedback

 ### For Administrators

1 . Access the admin dashboard using provided credentials

2 . View comprehensive candidate analytics

3 . Export candidate data in CSV or JSON format

4 . Track assessment metrics and skill distribution

 ## Technical Details

 ### Core Technologies

 -  * *Frontend * *: Streamlit

 -  * *Backend * *: Python

 -  * *AI Model * *: Google Gemini-1.5-pro

- **Vector Store**: ChromaDB

- **Embeddings**: HuggingFace's all-MiniLM-L6-v2

### Architecture Components

1. **Input Processing**

- File validation for resumes

- Data validation for user inputs

- Secure storage of candidate information

2. **Question Generation**

- Dynamic prompting based on skill levels

- Mix of multiple-choice and text-based questions

- Automated scoring and feedback

3. **Vector Database**

- Document storage and retrieval

- Efficient similarity search

- Persistent storage for scalability

## Prompt Design

### Information Gathering

The system uses a stage-based approach for gathering information:

```python

stages = {

"upload _resume": {...},

"name": {...},

"email": {...},

 # Additional stages

}

```

### Technical Question Generation

Questions are generated using carefully crafted prompts:

```python

prompt = f"""Generate 2 multiple choice questions and 1 text-based question for {tech} at proficiency level {level}/10.

Format for MCQs:

MCQ||||||

Format for text questions:

TEXT||"""
```


## Challenges & Solutions

### 1. Dynamic Question Generation

**Challenge**: Ensuring consistent quality and relevance of generated questions.

**Solution**: Implemented structured prompts with specific formats and validation checks.

### 2. Scalability

**Challenge**: Handling multiple candidates and large resume files efficiently.

**Solution**: Utilized ChromaDB for vector storage and implemented proper file size limits.

### 3. Assessment Accuracy

**Challenge**: Fair evaluation of text-based answers.

**Solution**: Developed a robust scoring system with answer normalization and pattern matching.

### 4. Security

**Challenge**: Protecting sensitive candidate data and preventing unauthorized access.

**Solution**:

- Implemented secure file validation

- Added session management

- Rate limiting for login attempts

- Secure credential storage

## Future Enhancements

1. Automated resume parsing using NLP

2. Soft skills and personality profiling

3. Multi-language support

4. Enhanced recruiter dashboard with AI-powered insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
