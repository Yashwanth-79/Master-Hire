__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import pandas as pd
from datetime import datetime
import re

# Load environment variables and configure
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Constants
CANDIDATE_DATA_FILE = "candidate_data.json"
ADMIN_CREDENTIALS = {"admin": "admin123"}  # In production, use secure password storage
ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# Initialize LangChain and FAISS
VECTOR_DB_DIR = "faiss_storage"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Create or load FAISS vector store
if os.path.exists(os.path.join(VECTOR_DB_DIR, "index.faiss")):
    vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings)
else:
    # Create with placeholder text that will be updated with real data later
    vector_db = FAISS.from_texts(["TalentScout initial index"], embeddings)
    vector_db.save_local(VECTOR_DB_DIR)

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format"""
    digits = ''.join(filter(str.isdigit, phone))
    return 10 <= len(digits) <= 15

def validate_file(uploaded_file):
    """Validate uploaded resume file"""
    if not uploaded_file:
        return False, "No file uploaded"
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
    
    return True, "File valid"

def init_session_state():
    """Initialize session state variables with default values"""
    defaults = {
        "page": "login",
        "messages": [],
        "current_stage": "upload_resume",
        "candidate_details": {},
        "assessment_started": False,
        "current_question": 0,
        "questions": [],
        "results": {"mcq": [], "text": []},
        "resume": None,
        "login_attempts": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_candidate_data(data):
    """Save candidate data to JSON file with error handling"""
    try:
        existing_data = []
        if os.path.exists(CANDIDATE_DATA_FILE):
            with open(CANDIDATE_DATA_FILE, 'r') as f:
                existing_data = json.load(f)
        
        # Add timestamp and unique identifier
        data['submission_timestamp'] = datetime.now().isoformat()
        data['candidate_id'] = f"CAND_{len(existing_data) + 1}"
        
        existing_data.append(data)
        with open(CANDIDATE_DATA_FILE, 'w') as f:
            json.dump(existing_data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving candidate data: {str(e)}")
        return False

def generate_questions(tech_stack):
    """Generate technical questions based on tech stack with improved prompting"""
    questions = []
    for tech, level in tech_stack.items():
        prompt = f"""Generate 2 multiple choice questions and 1 text-based question for {tech} at proficiency level {level}/10.
        Questions should be clear, specific, and appropriate for the skill level.
        
        Format for MCQs:
        MCQ|<question>|<option1>|<option2>|<option3>|<option4>|<correct_answer>
        
        Format for text questions:
        TEXT|<question>|<correct_answer>
        
        Ensure questions test practical knowledge and problem-solving abilities."""
        
        try:
            response = model.generate_content(prompt)
            generated_questions = [q.strip() for q in response.text.split("\n") if q.strip()]
            
            # Validate question format
            for q in generated_questions:
                if q.startswith("MCQ") and len(q.split("|")) == 7:
                    questions.append(q)
                elif q.startswith("TEXT") and len(q.split("|")) == 3:
                    questions.append(q)
        except Exception as e:
            st.error(f"Error generating questions for {tech}: {str(e)}")
            logging.exception("Exception occurred while generating questions")
    
    return questions

def calculate_score(results):
    """Calculate assessment score with detailed breakdown"""
    mcq_score = sum(1 for q in results['mcq'] if q['is_correct'])
    text_score = sum(1 for q in results['text'] if q['is_correct'])
    total_questions = len(results['mcq']) + len(results['text'])
    
    if total_questions == 0:
        return 0, {"mcq": 0, "text": 0}
    
    total_score = ((mcq_score + text_score) / total_questions) * 100
    breakdown = {
        "mcq": (mcq_score / len(results['mcq'])) * 100 if results['mcq'] else 0,
        "text": (text_score / len(results['text'])) * 100 if results['text'] else 0
    }
    
    return total_score, breakdown

def store_input(stage_key, input_value):
    """Safely store user input in session state"""
    if stage_key not in st.session_state.candidate_details:
        st.session_state.candidate_details[stage_key] = input_value

def process_user_input(user_input):
    """Process user input with improved validation, error handling, and logical sequence"""
    if user_input.lower() in ["bye", "goodbye", "exit"]:
        return "end_conversation"

    stages = {
        "upload_resume": {
            "next_stage": "name",
            "response": "Please upload your resume.",
            "validation": lambda x: validate_file(x)[0],
            "error": "Please upload a valid resume file.",
            "store_key": "resume"
        },
        "name": {
            "next_stage": "email",
            "response": "Please provide your full name:",
            "validation": lambda x: bool(x.strip()) and len(x.strip().split()) >= 2,
            "error": "Please enter your full name (first and last name).",
            "store_key": "full_name"
        },
        "email": {
            "next_stage": "phone",
            "response": "Please provide your email address:",
            "validation": validate_email,
            "error": "Please enter a valid email address.",
            "store_key": "email"
        },
        "phone": {
            "next_stage": "experience",
            "response": "Please provide your phone number:",
            "validation": validate_phone,
            "error": "Please enter a valid phone number.",
            "store_key": "phone"
        },
        "experience": {
            "next_stage": "position",
            "response": "How many years of experience do you have?",
            "validation": lambda x: x.replace('.', '').isdigit() and 0 <= float(x) <= 50,
            "error": "Please enter a valid number of years (0-50).",
            "store_key": "experience"
        },
        "position": {
            "next_stage": "location",
            "response": "What position(s) are you interested in?",
            "validation": lambda x: len(x.strip()) >= 3,
            "error": "Please enter at least one position.",
            "store_key": "position"
        },
        "location": {
            "next_stage": "tech_stack",
            "response": "What's your current location?",
            "validation": lambda x: len(x.strip()) >= 3,
            "error": "Please enter a valid location.",
            "store_key": "location"
        },
        "tech_stack": {
            "next_stage": None,
            "response": "Please list your technical skills (comma-separated):",
            "validation": lambda x: len(x.strip().split(',')) >= 1,
            "error": "Please enter at least one technical skill.",
            "store_key": "tech_stack"
        }
    }

    current_stage = st.session_state.get('current_stage', 'upload_resume')
    if current_stage in stages:
        stage_info = stages[current_stage]
        
        # Special handling for resume upload stage
        if current_stage == "upload_resume":
            if not st.session_state.resume:
                return stage_info["response"]
            else:
                st.session_state.current_stage = stage_info["next_stage"]
                return stages[stage_info["next_stage"]]["response"]

        try:
            # Input validation
            if not stage_info["validation"](user_input):
                st.error(stage_info["error"])
                return None

            # Store valid input
            if stage_info["store_key"]:
                store_input(stage_info["store_key"], user_input)

            # Handle tech stack stage specially
            if current_stage == "tech_stack":
                tech_skills = [skill.strip() for skill in user_input.split(',') if skill.strip()]
                st.session_state.tech_stack = tech_skills
                st.session_state.current_stage = "ratings"
                return "collect_ratings"

            # Move to next stage
            next_stage = stage_info["next_stage"]
            if next_stage:
                st.session_state.current_stage = next_stage
                return stages[next_stage]["response"]

        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            return None

    return user_input

def display_results():
    """Display assessment results and handle completion"""
    score, breakdown = calculate_score(st.session_state.results)
    st.session_state.candidate_details['assessment_score'] = score
    st.session_state.candidate_details['score_breakdown'] = breakdown
    
    # Create a container for results
    results_container = st.container()
    
    with results_container:
        st.markdown("### Assessment Results")
        
        # Overall score with color coding
        score_color = "green" if score >= 70 else "orange" if score >= 50 else "red"
        st.markdown(f"**Overall Score:** <span style='color:{score_color}'>{score:.1f}%</span>", unsafe_allow_html=True)
        
        # Breakdown of scores
        st.markdown("#### Score Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Multiple Choice Questions", f"{breakdown['mcq']:.1f}%")
        with col2:
            st.metric("Technical Questions", f"{breakdown['text']:.1f}%")
        
        # Detailed results
        if st.session_state.results['mcq']:
            st.markdown("#### Multiple Choice Questions")
            for idx, result in enumerate(st.session_state.results['mcq'], 1):
                with st.expander(f"Question {idx}"):
                    st.write("**Question:** ", result['question'])
                    st.write("**Your Answer:** ", result['selected'])
                    st.write("**Correct Answer:** ", result['correct'])
                    if result['is_correct']:
                        st.success("Correct! ✓")
                    else:
                        st.error("Incorrect ✗")
        
        if st.session_state.results['text']:
            st.markdown("#### Technical Questions")
            for idx, result in enumerate(st.session_state.results['text'], 1):
                with st.expander(f"Question {idx}"):
                    st.write("**Question:** ", result['question'])
                    st.write("**Your Answer:** ", result['answer'])
                    st.write("**Expected Answer:** ", result['correct'])
                    if result['is_correct']:
                        st.success("Correct! ✓")
                    else:
                        st.error("Incorrect ✗")
        
        # Save results
        if save_candidate_data(st.session_state.candidate_details):
            st.success("Your assessment results have been saved successfully!")
            
            # Feedback based on score
            if score >= 70:
                st.markdown("🌟 **Excellent performance!** Our team will be in touch with you soon.")
            elif score >= 50:
                st.markdown("👍 **Good effort!** We appreciate your participation.")
            else:
                st.markdown("📚 **Thank you for participating.** Keep learning and improving!")
            
            if st.button("Finish Assessment", type="primary"):
                st.session_state.page = "login"
                st.experimental_rerun()
        else:
            st.error("There was an error saving your results. Please contact support.")

def render_tech_stack_ratings():
    """Render interface for rating technical skills"""
    st.subheader("Rate your proficiency in each technology (1-10):")
    
    ratings = {}
    cols = st.columns(2)
    for idx, tech in enumerate(st.session_state.tech_stack):
        with cols[idx % 2]:
            ratings[tech] = st.slider(
                f"{tech}",
                min_value=1,
                max_value=10,
                value=5,
                help=f"Rate your proficiency in {tech} from 1 (beginner) to 10 (expert)"
            )

    if st.button("Start Assessment", type="primary"):
        st.session_state.tech_stack_ratings = ratings
        with st.spinner("Generating assessment questions..."):
            st.session_state.questions = generate_questions(ratings)
        st.session_state.assessment_started = True
        st.experimental_rerun()

def render_question(question_data):
    """Render assessment questions with improved UI"""
    st.progress((st.session_state.current_question + 1) / len(st.session_state.questions))
    st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}")

    if question_data.startswith("MCQ"):
        _, question, *options, correct = question_data.split("|")
        st.markdown(f"### {question}")
        answer = st.radio("Select your answer:", options, key=f"mcq_{st.session_state.current_question}")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Next Question", type="primary"):
                st.session_state.results['mcq'].append({
                    "question": question,
                    "selected": answer,
                    "correct": correct,
                    "is_correct": answer == correct
                })
                st.session_state.current_question += 1
                st.experimental_rerun()
    
    elif question_data.startswith("TEXT"):
        _, question, correct = question_data.split("|")
        st.markdown(f"### {question}")
        answer = st.text_area("Your answer:", key=f"text_{st.session_state.current_question}")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Next Question", type="primary"):
                st.session_state.results['text'].append({
                    "question": question,
                    "answer": answer,
                    "correct": correct,
                    "is_correct": answer.lower().strip() == correct.lower().strip()
                })
                st.session_state.current_question += 1
                st.experimental_rerun()

def chat_interface():
    """Enhanced chat interface with improved flow control"""
    st.title("   TalentScout Hiring Assistant🧑‍💻")

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).write(message['content'])

    # Handle resume upload stage
    if st.session_state.current_stage == "upload_resume":
        resume_file = st.file_uploader(
            "Upload your resume (PDF, DOC, DOCX):",
            type=['pdf', 'doc', 'docx'],
            help="Maximum file size: 200MB"
        )
        if resume_file is not None:
            is_valid, message = validate_file(resume_file)
            if is_valid:
                st.session_state.resume = resume_file
                st.session_state.current_stage = "name"
                st.success("Resume uploaded successfully!")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "What's your full name?"
                })
                st.experimental_rerun()
            else:
                st.error(message)

    # Handle assessment stage
    elif st.session_state.assessment_started:
        if st.session_state.current_question < len(st.session_state.questions):
            render_question(st.session_state.questions[st.session_state.current_question])
        else:
            display_results()

    # Handle tech stack ratings
    elif st.session_state.current_stage == "ratings":
        render_tech_stack_ratings()

    # Handle normal chat input
    else:
        user_input = st.chat_input("Type your message...")
        if user_input:
            response = process_user_input(user_input)
            if response:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.experimental_rerun()

def admin_dashboard():
    """Enhanced admin dashboard with additional features"""
    st.title("Admin Dashboard")
    
    # Sidebar with admin controls
    st.sidebar.title("Admin Controls")
    if st.sidebar.button("Logout"):
        st.session_state.page = "login"
        st.experimental_rerun()

    # Main dashboard content
    if os.path.exists(CANDIDATE_DATA_FILE):
        with open(CANDIDATE_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        if not data:
            st.info("No candidate data available yet.")
            return

        df = pd.DataFrame(data)
        
        # Dashboard metrics
        st.subheader("Dashboard Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", len(df))
        with col2:
            avg_score = df['assessment_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col3:
            recent_candidates = len(df[pd.to_datetime(df['submission_timestamp']) > 
                                    pd.Timestamp.now() - pd.Timedelta(days=7)])
            st.metric("New This Week", recent_candidates)

        # Candidate data analysis
        st.subheader("Candidate Analysis")
        tab1, tab2, tab3 = st.tabs(["Candidates", "Skills", "Export"])

        with tab1:
            # Search and filter
            search = st.text_input("Search candidates by name or email:")
            if search:
                mask = df['full_name'].str.contains(search, case=False, na=False) | \
                       df['email'].str.contains(search, case=False, na=False)
                filtered_df = df[mask]
            else:
                filtered_df = df

            # Display candidate data
            st.dataframe(
                filtered_df[[
                    'candidate_id', 'full_name', 'email', 'position',
                    'experience', 'assessment_score', 'submission_timestamp'
                ]],
                hide_index=True
            )

        with tab2:
            # Skills analysis
            st.write("Top Technical Skills")
            all_skills = []
            skill_ratings = {}
            
            for candidate in data:
                if 'tech_stack_ratings' in candidate:
                    for skill, rating in candidate['tech_stack_ratings'].items():
                        all_skills.append(skill)
                        if skill not in skill_ratings:
                            skill_ratings[skill] = []
                        skill_ratings[skill].append(rating)
            
            if all_skills:
                skill_df = pd.DataFrame({
                    'Skill': list(skill_ratings.keys()),
                    'Average Rating': [sum(ratings)/len(ratings) 
                                     for ratings in skill_ratings.values()],
                    'Candidates': [len(ratings) for ratings in skill_ratings.values()]
                })
                skill_df = skill_df.sort_values('Candidates', ascending=False)
                st.dataframe(skill_df, hide_index=True)
            else:
                st.info("No skill data available yet.")

        with tab3:
            # Export options
            st.write("Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if st.download_button(
                    "Download Full Report (CSV)",
                    df.to_csv(index=False),
                    "candidate_report.csv",
                    "text/csv"
                ):
                    st.success("Report downloaded successfully!")
            
            with col2:
                if st.download_button(
                    "Download Raw Data (JSON)",
                    json.dumps(data, indent=4),
                    "candidate_data.json",
                    "application/json"
                ):
                    st.success("Raw data downloaded successfully!")
    else:
        st.info("No candidate data available yet.")

def main():
    """Main application with improved security and error handling"""
    init_session_state()

    # Handle session timeout (30 minutes)
    if 'last_activity' in st.session_state:
        if (datetime.now() - st.session_state.last_activity).seconds > 1800:  # 30 minutes
            st.session_state.page = "login"
            st.warning("Session timed out. Please log in again.")
    st.session_state.last_activity = datetime.now()

    if st.session_state.page == "login":
        st.title("Welcome to TalentScout")
        
        # Center the login options
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                ### Choose Login Type
                Select how you want to proceed:
            """)
            
            if st.button("🧑‍💼 Candidate Login", use_container_width=True):
                st.session_state.page = "chat"
                st.session_state.messages = [{"role": "assistant", "content": "Welcome! I'm Master Hire, I am here to assist in you in hiring process Let's start with your application. "}]
                st.experimental_rerun()
            
            st.markdown("---")
            
            st.markdown("### Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("🔐 Login as Admin", use_container_width=True):
                if st.session_state.login_attempts >= 3:
                    st.error("Too many login attempts. Please try again later.")
                     # Add delay to prevent brute force
                elif ADMIN_CREDENTIALS.get(username) == password:
                    st.session_state.page = "admin"
                    st.session_state.login_attempts = 0
                    st.experimental_rerun()
                else:
                    st.session_state.login_attempts += 1
                    st.error("Invalid credentials")

    elif st.session_state.page == "chat":
        try:
            chat_interface()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.button("Reset Application", on_click=init_session_state)

    elif st.session_state.page == "admin":
        try:
            admin_dashboard()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.button("Return to Login"):
                st.session_state.page = "login"
                st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.button("Restart Application", on_click=init_session_state)
