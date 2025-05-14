import streamlit as st
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import re
import logging
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")

st.set_page_config(page_title="CV Question Generator", page_icon="📄", layout="centered")

def extract_cv_text(pdf_file):
    """Extract text from a PDF CV with improved handling and OCR fallback suggestion."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # Remove common garbled characters
                page_text = re.sub(r'\(cid:\d+\)', '', page_text)
                text += page_text
        logger.debug(f"Extracted text: {text[:500]}...")
        return text.strip() or "No readable text extracted (likely scanned PDF - use OCR)"
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}. The CV may be a scanned PDF - use OCR (e.g., Adobe Acrobat, Tesseract).")
        return ""

def preprocess_cv(cv_text):
    """Split CV text into sentences and clean, handling bullet points."""
    try:
        cv_text = re.sub(r'•|\-|\*', '\n', cv_text)
        sentences = []
        for line in cv_text.split('\n'):
            line = line.strip()
            if len(line) > 3:
                line_sentences = sent_tokenize(line)
                sentences.extend(line_sentences if line_sentences else [line])
        cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        logger.debug(f"Preprocessed sentences: {len(cleaned_sentences)} sentences")
        return cleaned_sentences
    except LookupError as e:
        st.error(f"NLTK resource error: {str(e)}. Please run `nltk.download('punkt_tab')`.")
        return []

def detect_sections(cv_text):
    """Detect CV sections with broader keywords."""
    sections = {
        "Experience": [], "Education": [], "Skills": [], "Projects": [],
        "Associative": [], "Languages": [], "Personal": [], "Hobbies": [], "Certifications": [], "Other": []
    }
    current_section = "Other"
    lines = cv_text.split("\n")
    
    section_keywords = {
        "Experience": [r"\b(professional|work)?\s*(experience|history)\b", r"\bintern(ship)?\b", r"\bemployment\b", r"\bcareer\b", r"\bpositions?\b"],
        "Education": [r"\beducation\b", r"\bacademic background\b", r"\bdegree(s)?\b", r"\buniversity\b", r"\bstudies\b", r"\bqualification(s)?\b"],
        "Skills": [r"\bskills\b", r"\btechnical skills\b", r"\bproficiencies\b", r"\btechnologies\b", r"\bexpertise\b", r"\babilities\b", r"\btechniques\b"],
        "Projects": [r"\bprojects?\b", r"\bdeveloped a\b", r"\bsoftware\b", r"\bapplication(s)?\b", r"\bachievements?\b", r"\bportfolio\b"],
        "Associative": [r"\bassociative experience\b", r"\bclub(s)?\b", r"\borganization(s)?\b", r"\bvolunteer\b", r"\bextracurricular\b", r"\bactivities\b"],
        "Languages": [r"\blangues\b", r"\blanguages\b"],
        "Personal": [r"\bpersonnelles\b", r"\bpersonal\b", r"\bcontact\b", r"\binfo\b"],
        "Hobbies": [r"\bhobbies\b", r"\binterests\b", r"\bdiving\b"],
        "Certifications": [r"\bcertifications\b", r"\bcertificates\b", r"\btraining\b"]
    }
    
    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue
        for section, keywords in section_keywords.items():
            if any(re.search(keyword, line_lower) for keyword in keywords):
                current_section = section
                logger.debug(f"Detected section: {section} for line: {line_lower}")
                break
        sections[current_section].append(line.strip())
    
    logger.debug(f"Sections: { {k: len(v) for k, v in sections.items()} }")
    return sections

def extract_entities(sentence, section):
    """Extract named entities with enhanced detection for technical details."""
    try:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        tree = ne_chunk(pos_tags)
        entities = {
            "PERSON": [], "ORGANIZATION": [], "DATE": [], "ROLE": [], "PROJECT": [],
            "SKILL": [], "DEGREE": [], "EVENT": [], "LANGUAGE": [], "HOBBY": [],
            "CERTIFICATION": [], "METHODOLOGY": [], "DOMAIN": []
        }
        
        # Extract persons and organizations
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity = " ".join([word for word, tag in subtree.leaves()])
                if subtree.label() == "PERSON":
                    entities["PERSON"].append(entity)
                elif subtree.label() == "ORGANIZATION":
                    entities["ORGANIZATION"].append(entity)
                elif subtree.label() == "GPE":
                    if entity.lower() not in ["sfax", "tunis", "tunisia"]:
                        entities["ORGANIZATION"].append(entity)
        
        # Extract dates
        date_patterns = r"\b(19|20)\d{2}\s*[–-]\s*(19|20)\d{2}\b|\b(19|20)\d{2}\b"
        dates = re.findall(date_patterns, sentence)
        entities["DATE"] = [d[0] for d in dates] if dates else []
        
        # Extract roles
        role_patterns = r"\b([A-Z][a-zA-Z\s]*(?:Engineer|Developer|Intern|Manager|Lead|Coordinator|Analyst|Consultant|Designer|Stagiaire|Employee|Officer|Specialist))\b"
        roles = re.findall(role_patterns, sentence)
        if section in ["Experience", "Associative"] or any(r.lower() in sentence.lower() for r in ["intern", "stagiaire", "role", "position", "job"]):
            entities["ROLE"] = roles
        
        # Extract projects
        project_patterns = r"\b([A-Z][a-zA-Z\s]+(?:System|Project|Portal|Application|Software|App|Platform|Tool))\b"
        projects = re.findall(project_patterns, sentence)
        if section in ["Projects", "Experience"] or any(keyword in sentence.lower() for keyword in ["project", "developed", "built", "worked on", "created", "designed"]):
            entities["PROJECT"] = projects
        
        # Extract technical skills (expanded list)
        skill_patterns = r"\b(JavaScript|React|Django|PostgreSQL|Python|Java|SQL|Flask|Angular|Machine Learning|Deep Learning|Git|Docker|Kubernetes|AWS|Azure|GCP|Tailwind|TailwindCSS|Bootstrap|Frontend|Backend|HTML|CSS|CRUD|Node\.js|Express\.js|MongoDB|MySQL|Oracle|Redis|GraphQL|REST|API|TensorFlow|PyTorch|Scikit-learn|Linux|Jenkins|Ansible|Terraform|CI/CD|DevOps|[A-Z][a-zA-Z]+)\b"
        skills = re.findall(skill_patterns, sentence)
        if section == "Skills" or any(keyword in sentence.lower() for keyword in ["skills", "technologies", "proficient", "expertise", "tools", "knowledge", "trained", "stack"]):
            entities["SKILL"] = [skill for skill in skills if skill.lower() not in ["and", "or", "with", "at", "the", "of", "techniques", "personnelles", "langues", "stage", "gestion", "environnements", "word", "anglais", "diving", "sfax", "crud", "frontend", "backend"]]
        
        # Extract degrees
        degree_patterns = r"\b(Bachelor|Master|PhD|Degree|Diploma|Certificate)\s*(?:of\s*[A-Za-z\s]+)?\b"
        degrees = re.findall(degree_patterns, sentence)
        if section == "Education" or any(keyword in sentence.lower() for keyword in ["degree", "education", "qualification", "certification"]):
            entities["DEGREE"] = degrees
        
        # Extract events
        event_patterns = r"\b([A-Z][a-zA-Z\s]*(?:Workshop|Event|Activity|Program|Conference|Training))\b"
        events = re.findall(event_patterns, sentence)
        if section == "Associative":
            entities["EVENT"] = events
        
        # Extract languages
        language_patterns = r"\b(English|French|Arabic|Anglais|Français|Arabe|Spanish|German|Italian|Russian)\b"
        languages = re.findall(language_patterns, sentence, re.IGNORECASE)
        if section == "Languages" or "language" in sentence.lower():
            entities["LANGUAGE"] = languages
        
        # Extract hobbies
        hobby_patterns = r"\b([A-Z][a-zA-Z\s]*(?:Diving|Reading|Traveling|Photography|Gaming|Cooking|Hiking))\b"
        hobbies = re.findall(hobby_patterns, sentence)
        if section == "Hobbies" or "hobby" in sentence.lower():
            entities["HOBBY"] = hobbies
        
        # Extract certifications
        cert_patterns = r"\b([A-Z][a-zA-Z\s]*(?:Certified|Certification|Professional|AWS|Azure|Google|Cisco|Oracle|Scrum|ITIL|PMP))\b"
        certs = re.findall(cert_patterns, sentence)
        if section == "Certifications" or any(keyword in sentence.lower() for keyword in ["certification", "certificate", "certified", "training"]):
            entities["CERTIFICATION"] = certs
        
        # Extract methodologies
        method_patterns = r"\b(Agile|Scrum|Kanban|Waterfall|DevOps|CI/CD|TDD|BDD|OOP|Microservices|SOA)\b"
        methods = re.findall(method_patterns, sentence, re.IGNORECASE)
        if any(keyword in sentence.lower() for keyword in ["methodology", "approach", "practice", "workflow", "development process"]):
            entities["METHODOLOGY"] = methods
        
        # Extract technical domains
        domain_patterns = r"\b(Machine Learning|Artificial Intelligence|Data Science|Cybersecurity|Cloud Computing|Web Development|Mobile Development|Blockchain|IoT|Big Data|Networking|Database Management)\b"
        domains = re.findall(domain_patterns, sentence, re.IGNORECASE)
        if any(keyword in sentence.lower() for keyword in ["specialization", "domain", "field", "area of expertise"]):
            entities["DOMAIN"] = domains
        
        logger.debug(f"Entities for sentence '{sentence}' in section '{section}': {entities}")
        return entities
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        return {
            "PERSON": [], "ORGANIZATION": [], "DATE": [], "ROLE": [], "PROJECT": [],
            "SKILL": [], "DEGREE": [], "EVENT": [], "LANGUAGE": [], "HOBBY": [],
            "CERTIFICATION": [], "METHODOLOGY": [], "DOMAIN": []
        }

def collect_entities(cv_text):
    """Collect all entities from the CV for question generation."""
    sections = detect_sections(cv_text)
    all_entities = {
        "PERSON": set(), "ORGANIZATION": set(), "ROLE": set(), "PROJECT": set(),
        "SKILL": set(), "DEGREE": set(), "EVENT": set(), "DATE": set(), "LANGUAGE": set(),
        "HOBBY": set(), "CERTIFICATION": set(), "METHODOLOGY": set(), "DOMAIN": set()
    }
    
    for section, lines in sections.items():
        for sentence in preprocess_cv("\n".join(lines)):
            entities = extract_entities(sentence, section)
            for key in all_entities:
                all_entities[key].update(entities[key])
    
    # Filter out invalid entities
    all_entities["PERSON"] = set(p for p in all_entities["PERSON"] if p not in ["Sfax"])
    all_entities["SKILL"] = set(s for s in all_entities["SKILL"] if s.lower() not in ["techniques", "langues", "personnelles", "stage", "gestion", "environnements", "word", "anglais", "diving", "sfax", "crud", "frontend", "backend"])
    all_entities["PROJECT"] = set(p for p in all_entities["PROJECT"] if p.lower() not in ["techniques", "langues", "personnelles", "stage", "gestion", "environnements", "word", "anglais", "diving", "sfax", "crud", "frontend"])
    all_entities["ROLE"] = set(r for r in all_entities["ROLE"] if r.lower() not in ["techniques", "langues", "personnelles", "stage", "gestion", "environnements", "word", "anglais", "diving", "sfax", "crud", "frontend"])
    all_entities["CERTIFICATION"] = set(c for c in all_entities["CERTIFICATION"] if c.lower() not in ["techniques", "langues", "personnelles", "stage", "gestion", "environnements", "word", "anglais", "diving", "sfax", "crud", "frontend"])
    
    logger.debug(f"All extracted entities: {all_entities}")
    return all_entities

def generate_technical_mc_questions(all_entities, cv_text, person):
    """Generate technical multiple-choice questions based on CV entities or roles."""
    technical_questions = []
    
    # Skill-based technical questions
    for skill in all_entities["SKILL"]:
        org = next((org for org in all_entities["ORGANIZATION"] if skill in cv_text and org in cv_text), "their organization")
        distractors = random.sample(["Ruby", "C#", "PHP", "Go", "Swift", "Kotlin", "Rust", "Scala"], 4)
        options = [skill] + distractors[:3]
        random.shuffle(options)
        technical_questions.append({
            "question": f"Which technology did {person} work with at {org}?",
            "options": options,
            "answer": skill
        })
    
    # Project-based technical questions
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        skill = next((skill for skill in all_entities["SKILL"] if skill in cv_text), "a technology")
        distractors = random.sample(["Performance Optimization", "Security Implementation", "Database Integration", "API Development"], 3)
        options = [f"Development with {skill}"] + distractors
        random.shuffle(options)
        technical_questions.append({
            "question": f"What technical aspect did {person} focus on while working on {project} at {org}?",
            "options": options,
            "answer": f"Development with {skill}"
        })
    
    # Certification-based technical questions
    for cert in all_entities["CERTIFICATION"]:
        distractors = random.sample(["AWS Solutions Architect", "Cisco CCNA", "Google Professional Data Engineer", "Scrum Master"], 3)
        options = [cert] + distractors
        random.shuffle(options)
        technical_questions.append({
            "question": f"Which technical certification does {person} hold?",
            "options": options,
            "answer": cert
        })
    
    # Methodology-based technical questions
    for method in all_entities["METHODOLOGY"]:
        distractors = random.sample(["Waterfall", "Kanban", "Scrum", "TDD"], 3)
        options = [method] + distractors
        random.shuffle(options)
        technical_questions.append({
            "question": f"Which development methodology has {person} worked with?",
            "options": options,
            "answer": method
        })
    
    # Domain-based technical questions
    for domain in all_entities["DOMAIN"]:
        distractors = random.sample(["Cybersecurity", "Web Development", "Blockchain", "IoT"], 3)
        options = [domain] + distractors
        random.shuffle(options)
        technical_questions.append({
            "question": f"Which technical domain has {person} specialized in?",
            "options": options,
            "answer": domain
        })
    
    # Role-based technical questions (enhanced experience-based)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        if any(r.lower() in role.lower() for r in ["engineer", "developer", "analyst", "intern", "stagiaire", "consultant", "specialist"]):
            # Technical task question
            distractors = random.sample(["Manual Testing", "Documentation Writing", "Stakeholder Meetings", "Budget Planning"], 3)
            options = ["Code Optimization"] + distractors
            random.shuffle(options)
            technical_questions.append({
                "question": f"As a {role} at {org}, what technical task did {person} likely perform?",
                "options": options,
                "answer": "Code Optimization"
            })
            # Debugging question
            distractors = random.sample(["Customer Support", "Marketing Campaigns", "Financial Reporting", "Event Planning"], 3)
            options = ["Debugging Issues"] + distractors
            random.shuffle(options)
            technical_questions.append({
                "question": f"What technical responsibility did {person} have as a {role} at {org}?",
                "options": options,
                "answer": "Debugging Issues"
            })
            # System design question
            distractors = random.sample(["User Interface Design", "Sales Strategy", "HR Policies", "Logistics Planning"], 3)
            options = ["System Architecture Design"] + distractors
            random.shuffle(options)
            technical_questions.append({
                "question": f"What technical aspect of system design did {person} contribute to as a {role} at {org}?",
                "options": options,
                "answer": "System Architecture Design"
            })
            # Collaboration question
            distractors = random.sample(["Team Building", "Marketing Collaboration", "Financial Audits", "Event Coordination"], 3)
            options = ["Technical Team Collaboration"] + distractors
            random.shuffle(options)
            technical_questions.append({
                "question": f"What type of technical collaboration did {person} engage in as a {role} at {org}?",
                "options": options,
                "answer": "Technical Team Collaboration"
            })
    
    return technical_questions

def generate_technical_tf_questions(all_entities, cv_text, person):
    """Generate technical true/false questions based on CV entities or roles."""
    technical_questions = []
    
    # Skill-based technical questions
    for skill in all_entities["SKILL"]:
        org = next((org for org in all_entities["ORGANIZATION"] if skill in cv_text and org in cv_text), "their organization")
        technical_questions.append({
            "question": f"Did {person} use {skill} at {org}?",
            "answer": True
        })
    
    # Project-based technical questions
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        skill = next((skill for skill in all_entities["SKILL"] if skill in cv_text), "a technology")
        technical_questions.append({
            "question": f"Did {person} use {skill} while working on {project} at {org}?",
            "answer": True
        })
    
    # Certification-based technical questions
    for cert in all_entities["CERTIFICATION"]:
        technical_questions.append({
            "question": f"Does {person} hold a {cert} certification?",
            "answer": True
        })
    
    # Methodology-based technical questions
    for method in all_entities["METHODOLOGY"]:
        technical_questions.append({
            "question": f"Has {person} used the {method} methodology in their work?",
            "answer": True
        })
    
    # Domain-based technical questions
    for domain in all_entities["DOMAIN"]:
        technical_questions.append({
            "question": f"Has {person} worked in the {domain} field?",
            "answer": True
        })
    
    # Role-based technical questions (enhanced experience-based)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        if any(r.lower() in role.lower() for r in ["engineer", "developer", "analyst", "intern", "stagiaire", "consultant", "specialist"]):
            technical_questions.append({
                "question": f"Did {person} handle technical debugging as a {role} at {org}?",
                "answer": True
            })
            technical_questions.append({
                "question": f"Did {person} perform system optimization as a {role} at {org}?",
                "answer": True
            })
            technical_questions.append({
                "question": f"Did {person} contribute to system architecture design as a {role} at {org}?",
                "answer": True
            })
            technical_questions.append({
                "question": f"Did {person} collaborate with a technical team as a {role} at {org}?",
                "answer": True
            })
    
    return technical_questions

def generate_technical_open_questions(all_entities, cv_text, person):
    """Generate technical open-ended questions based on CV entities or roles."""
    technical_questions = []
    
    # Skill-based technical questions
    for skill in all_entities["SKILL"]:
        org = next((org for org in all_entities["ORGANIZATION"] if skill in cv_text and org in cv_text), "their organization")
        technical_questions.append(f"How has {person} applied {skill} in their role at {org}?")
    
    # Project-based technical questions
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        skill = next((skill for skill in all_entities["SKILL"] if skill in cv_text), "a technology")
        technical_questions.append(f"How did {person} use {skill} while working on {project} at {org}?")
    
    # Certification-based technical questions
    for cert in all_entities["CERTIFICATION"]:
        technical_questions.append(f"How has {person}'s {cert} certification impacted their technical work?")
    
    # Methodology-based technical questions
    for method in all_entities["METHODOLOGY"]:
        technical_questions.append(f"Can {person} describe their experience using the {method} methodology?")
    
    # Domain-based technical questions
    for domain in all_entities["DOMAIN"]:
        technical_questions.append(f"What challenges has {person} faced while working in {domain}?")
    
    # Role-based technical questions (enhanced experience-based)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        if any(r.lower() in role.lower() for r in ["engineer", "developer", "analyst", "intern", "stagiaire", "consultant", "specialist"]):
            technical_questions.append(f"What technical challenges did {person} face as a {role} at {org}?")
            technical_questions.append(f"How did {person} approach debugging technical issues as a {role} at {org}?")
            technical_questions.append(f"Can {person} describe a time they optimized a system or process as a {role} at {org}?")
            technical_questions.append(f"How did {person} contribute to system design as a {role} at {org}?")
            technical_questions.append(f"Can {person} share an example of technical collaboration as a {role} at {org}?")
    
    return technical_questions

def generate_multiple_choice_questions(cv_text, max_questions=10):
    """Generate multiple-choice questions with CV-specific technical questions."""
    all_entities = collect_entities(cv_text)
    questions = []
    has_technical_content = bool(all_entities["SKILL"] or all_entities["PROJECT"] or all_entities["CERTIFICATION"] or all_entities["METHODOLOGY"] or all_entities["DOMAIN"])
    
    person = next(iter(all_entities["PERSON"]), "the candidate")
    
    # Generate technical questions
    technical_questions = generate_technical_mc_questions(all_entities, cv_text, person)
    questions.extend(technical_questions)
    
    # Role-based questions (non-technical)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        distractors = random.sample(["Software Engineer", "Data Analyst", "Product Manager", "Designer", "Marketing Specialist", "HR Manager"], 4)
        options = [role] + distractors[:3]
        random.shuffle(options)
        questions.append({
            "question": f"What position did {person} hold at {org}?",
            "options": options,
            "answer": role
        })
    
    # Project-based questions (non-technical)
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        distractors = random.sample(["Inventory Management App", "E-commerce Platform", "Chat Application", "AI Recommendation System", "CRM System", "Payment Gateway"], 4)
        options = [project] + distractors[:3]
        random.shuffle(options)
        questions.append({
            "question": f"Which project did {person} contribute to at {org}?",
            "options": options,
            "answer": project
        })
    
    # Degree-based questions
    for degree in all_entities["DEGREE"]:
        distractors = random.sample(["Master of Science", "PhD in Engineering", "Diploma in Arts", "Bachelor of Business", "Certificate in IT"], 4)
        options = [degree] + distractors[:3]
        random.shuffle(options)
        questions.append({
            "question": f"What academic qualification does {person} have?",
            "options": options,
            "answer": degree
        })
    
    # Language-based questions
    for language in all_entities["LANGUAGE"]:
        distractors = random.sample(["Spanish", "German", "Italian", "Mandarin", "Russian", "Japanese"], 4)
        options = [language] + distractors[:3]
        random.shuffle(options)
        questions.append({
            "question": f"Which language is {person} proficient in?",
            "options": options,
            "answer": language
        })
    
    # Hobby-based questions
    for hobby in all_entities["HOBBY"]:
        distractors = random.sample(["Hiking", "Painting", "Gaming", "Cooking", "Photography", "Swimming"], 4)
        options = [hobby] + distractors[:3]
        random.shuffle(options)
        questions.append({
            "question": f"Which activity does {person} enjoy as a hobby?",
            "options": options,
            "answer": hobby
        })
    
    # General fallback questions (randomized, non-technical)
    fallback_questions = [
        {"question": f"What is {person}'s primary area of expertise?", "options": ["Programming", "Design", "Marketing", "Finance"], "answer": random.choice(["Programming", "Design", "Marketing"])},
        {"question": f"Has {person} worked in a technical field?", "options": ["Yes", "No"], "answer": "Yes" if has_technical_content else "No"},
        {"question": f"What type of project might {person} have worked on?", "options": ["Web App", "Mobile App", "Data Analysis", "Graphic Design"], "answer": random.choice(["Web App", "Mobile App", "Data Analysis"])},
        {"question": f"Does {person} have experience in management?", "options": ["Yes", "No"], "answer": "No"},
        {"question": f"Which industry might {person} be involved in?", "options": ["Tech", "Healthcare", "Education", "Retail"], "answer": random.choice(["Tech", "Healthcare"])}
    ]
    if len(questions) < max_questions:
        questions.extend([random.choice(fallback_questions) for _ in range(max_questions - len(questions))])
    
    return questions[:max_questions], bool(technical_questions)

def generate_true_false_questions(cv_text, max_questions=10):
    """Generate true/false questions with CV-specific technical questions."""
    all_entities = collect_entities(cv_text)
    questions = []
    has_technical_content = bool(all_entities["SKILL"] or all_entities["PROJECT"] or all_entities["CERTIFICATION"] or all_entities["METHODOLOGY"] or all_entities["DOMAIN"])
    
    person = next(iter(all_entities["PERSON"]), "the candidate")
    
    # Generate technical questions
    technical_questions = generate_technical_tf_questions(all_entities, cv_text, person)
    questions.extend(technical_questions)
    
    # Role-based questions (non-technical)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        questions.append({
            "question": f"Was {person} a {role} at {org}?",
            "answer": True
        })
    
    # Project-based questions (non-technical)
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        questions.append({
            "question": f"Did {person} work on {project} at {org}?",
            "answer": True
        })
    
    # Degree-based questions
    for degree in all_entities["DEGREE"]:
        questions.append({
            "question": f"Does {person} hold a {degree}?",
            "answer": True
        })
    
    # Language-based questions
    for language in all_entities["LANGUAGE"]:
        questions.append({
            "question": f"Is {person} fluent in {language}?",
            "answer": True
        })
    
    # Hobby-based questions
    for hobby in all_entities["HOBBY"]:
        questions.append({
            "question": f"Does {person} enjoy {hobby} as a hobby?",
            "answer": True
        })
    
    # General fallback questions (randomized, non-technical)
    fallback_questions = [
        {"question": f"Has {person} worked in a professional role?", "answer": True},
        {"question": f"Does {person} have technical skills?", "answer": True if has_technical_content else False},
        {"question": f"Has {person} completed a significant project?", "answer": True},
        {"question": f"Does {person} have leadership experience?", "answer": random.choice([True, False])},
        {"question": f"Has {person} worked abroad?", "answer": random.choice([True, False])}
    ]
    if len(questions) < max_questions:
        questions.extend([random.choice(fallback_questions) for _ in range(max_questions - len(questions))])
    
    return questions[:max_questions], bool(technical_questions)

def generate_open_questions(cv_text, max_questions=10):
    """Generate open-ended questions with CV-specific technical questions."""
    all_entities = collect_entities(cv_text)
    questions = []
    has_technical_content = bool(all_entities["SKILL"] or all_entities["PROJECT"] or all_entities["CERTIFICATION"] or all_entities["METHODOLOGY"] or all_entities["DOMAIN"])
    
    person = next(iter(all_entities["PERSON"]), "the candidate")
    
    # Generate technical questions
    technical_questions = generate_technical_open_questions(all_entities, cv_text, person)
    questions.extend(technical_questions)
    
    # Role-based questions (non-technical)
    for role in all_entities["ROLE"]:
        org = next((org for org in all_entities["ORGANIZATION"] if role in cv_text and org in cv_text), "their organization")
        questions.append(f"What were {person}'s key responsibilities as a {role} at {org}?")
    
    # Project-based questions (non-technical)
    for project in all_entities["PROJECT"]:
        org = next((org for org in all_entities["ORGANIZATION"] if project in cv_text and org in cv_text), "their organization")
        questions.append(f"Can {person} describe their contribution to {project} at {org}?")
    
    # Degree-based questions
    for degree in all_entities["DEGREE"]:
        questions.append(f"How has {person}'s {degree} influenced their career path?")
    
    # Language-based questions
    for language in all_entities["LANGUAGE"]:
        questions.append(f"How has {person}'s fluency in {language} been beneficial in their professional work?")
    
    # Hobby-based questions
    for hobby in all_entities["HOBBY"]:
        questions.append(f"How does {person} balance their interest in {hobby} with their professional responsibilities?")
    
    # General fallback questions (randomized, non-technical)
    fallback_questions = [
        f"Can {person} describe their most significant professional achievement?",
        f"What motivates {person} in their career?",
        f"How does {person} approach problem-solving in their work?",
        f"What challenges has {person} faced in their career?",
        f"How does {person} stay updated in their field?"
    ]
    if len(questions) < max_questions:
        questions.extend(random.sample(fallback_questions, max_questions - len(questions)))
    
    return questions[:max_questions], bool(technical_questions)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4d4d;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #cc0000;
    }
    .stFileUploader {
        background-color: #2a2a2a;
        border: 1px dashed #666;
        border-radius: 5px;
        padding: 10px;
    }
    .stRadio label {
        color: #ffffff;
    }
    .stSuccess {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #ffca2c;
        color: black;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 CV Question Generator")
st.write("Upload a CV (PDF) to generate interview questions.")

# Debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode (Show Extracted Text and Sections)")

# Question type selection
st.subheader("Choisissez un type de question :")
question_type = st.radio(
    "",
    ("Questions à choix multiples", "Questions vrai ou faux", "Questions ouvertes"),
    index=1  # Default to "Questions vrai ou faux"
)

# File uploader with drag-and-drop
uploaded_file = st.file_uploader("Choose a PDF CV", type=["pdf"], accept_multiple_files=False, help="Limit 200MB per file • PDF")
if uploaded_file:
    st.write(f"Selected file: {uploaded_file.name} ({uploaded_file.size / 1024:.1f}KB)")
    if st.button("Générer des questions"):
        with st.spinner("Processing CV..."):
            cv_text = extract_cv_text(uploaded_file)
            if cv_text.strip():
                if debug_mode:
                    st.subheader("Debug Information")
                    st.write("**Extracted Text (first 500 characters):**")
                    st.text(cv_text[:500] + "...")
                    sections = detect_sections(cv_text)
                    st.write("**Detected Sections:**")
                    for section, content in sections.items():
                        st.write(f"{section}: {len(content)} lines")
                        if content:
                            st.text("\n".join(content[:3]))
                
                if question_type == "Questions à choix multiples":
                    questions, has_technical_questions = generate_multiple_choice_questions(cv_text)
                    if questions:
                        st.success("Questions generated successfully!")
                        if not has_technical_questions:
                            st.warning("No specific technical details (e.g., skills, projects, certifications) were detected in the CV, but technical questions about experience have been generated based on roles.")
                        st.subheader("Generated Questions:")
                        for i, q in enumerate(questions, 1):
                            st.write(f"{i}. {q['question']}")
                            st.write("Options:", ", ".join(q['options']))
                            st.write(f"Answer: {q['answer']}")
                            st.write("---")
                    else:
                        st.warning("No multiple-choice questions generated. The CV may lack recognizable entities or be a scanned PDF. Convert to text-based PDF using OCR if scanned.")
                
                elif question_type == "Questions vrai ou faux":
                    questions, has_technical_questions = generate_true_false_questions(cv_text)
                    if questions:
                        st.success("Questions generated successfully!")
                        if not has_technical_questions:
                            st.warning("No specific technical details (e.g., skills, projects, certifications) were detected in the CV, but technical questions about experience have been generated based on roles.")
                        st.subheader("Generated Questions:")
                        for i, q in enumerate(questions, 1):
                            st.write(f"{i}. {q['question']}")
                            st.write(f"Answer: {'True' if q['answer'] else 'False'}")
                            st.write("---")
                    else:
                        st.warning("No true/false questions generated. The CV may lack recognizable entities or be a scanned PDF. Convert to text-based PDF using OCR if scanned.")
                
                elif question_type == "Questions ouvertes":
                    questions, has_technical_questions = generate_open_questions(cv_text)
                    if questions:
                        st.success("Questions generated successfully!")
                        if not has_technical_questions:
                            st.warning("No specific technical details (e.g., skills, projects, certifications) were detected in the CV, but technical questions about experience have been generated based on roles.")
                        st.subheader("Generated Questions:")
                        for i, q in enumerate(questions, 1):
                            st.write(f"{i}. {q}")
                            st.write("---")
                    else:
                        st.warning("No open-ended questions generated. The CV may lack recognizable entities or be a scanned PDF. Convert to text-based PDF using OCR if scanned.")
            else:
                st.error("No text extracted. The CV is likely a scanned PDF. Please convert it to a text-based PDF using OCR (e.g., Adobe Acrobat, Tesseract) and upload again.")
else:
    st.info("Drag and drop a PDF CV here or click 'Browse files' to start.")

st.markdown("---")
st.write("Built with Streamlit")