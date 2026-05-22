# app_with_models.py — ChatLaw Backend with AI Models (Stable Loading)
import os
import re
import uuid
import logging
import glob
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Logging
logger = logging.getLogger("chatlaw")
logging.basicConfig(level=logging.INFO)

# -------------------------
# Safe Model Loading
# -------------------------
embedder = None
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
ollama_available = False

def check_ollama():
    """Ping Ollama and confirm the model is available."""
    global ollama_available
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                ollama_available = True
                logger.info(f"✓ Ollama ready — model: {OLLAMA_MODEL}")
            else:
                logger.warning(f"Ollama running but '{OLLAMA_MODEL}' not found. Pull it with: ollama pull {OLLAMA_MODEL}")
        else:
            logger.warning("Ollama responded with unexpected status")
    except Exception as e:
        logger.warning(f"Ollama not reachable: {e}. Falling back to templates.")

def load_embedder():
    """Load sentence-transformers embedder."""
    global embedder
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        logger.info("✓ Embedding model loaded")
    except Exception as e:
        logger.warning(f"Embedding model failed: {e}")
        embedder = None

check_ollama()
load_embedder()

# Import other dependencies
try:
    import pandas as pd
    import numpy as np
    import faiss
    HAS_FAISS = True
except:
    HAS_FAISS = False
    logger.warning("FAISS not available, search disabled")

# -------------------------
# Pydantic models (API)
# -------------------------
class ConsultationStartRequest(BaseModel):
    query: str
    max_turns: Optional[int] = 7

class ConsultationStartResponse(BaseModel):
    session_id: str
    next_action: str
    question: Optional[str] = None
    partial_report: Optional[str] = None

class ConsultationAnswerRequest(BaseModel):
    session_id: str
    answer: str

class ConsultationFinalResponse(BaseModel):
    session_id: str
    report: str
    structured: Dict
    timestamp: str

# -------------------------
# Knowledge Graph
# -------------------------
class KnowledgeGraph:
    def __init__(self):
        self.entities = defaultdict(list)
        self.relations = []
        self.context = {}

    def add_entity(self, entity_type: str, value: str):
        if value and value.strip() and value not in self.entities[entity_type]:
            self.entities[entity_type].append(value.strip())

    def set_context(self, key: str, value: str):
        self.context[key] = value

    def get_summary(self) -> str:
        parts = []
        for entity_type, values in self.entities.items():
            if values:
                parts.append(f"{entity_type.upper()}: {', '.join(values[:3])}")
        if self.context.get('situation'):
            parts.append(f"SITUATION: {self.context['situation'][:200]}")
        return " | ".join(parts) if parts else "Knowledge graph empty"

# -------------------------
# Smart Classifier Agent
# -------------------------
class SmartClassifierAgent:
    def __init__(self):
        self.case_keywords = {
            'criminal': ['theft','stolen','robbery','assault','murder','rape','dacoity','fir','police','crime','burglar','pickpocket','extortion','blackmail','kidnap','abduct','arson','fraud','cheat','forgery','bribe','corruption'],
            'family': ['divorce','marriage','custody','alimony','maintenance','dowry','wife','husband','domestic','separation','child','guardian','adoption','succession','will','heir'],
            'property': ['land','boundary','inheritance','encroachment','tenant','landlord','eviction','deed','title','plot','mutation','rent','possession','trespass','partition'],
            'contract': ['agreement','breach','contract','payment','outstanding','invoice','debt','loan','delivery','default','guarantee','indemnity','arbitration'],
            'cyber': ['online','internet','hack','hacked','cyber','data','privacy','leak','leaked','screenshot','social media','whatsapp','facebook','instagram','twitter','email','photo','video','morphed','deepfake','troll','defam','otp','phishing','password','account','digital','private chat','chat','share','forward','viral','revenge porn','sextortion','cyberbully','bully','stalk','doxx','identity theft','it act'],
            'consumer': ['product','defective','refund','warranty','seller','buyer','purchase','return','complaint','shop','shopping','delivery','service','manufacturer','consumer','overcharg','mislead','adulterat','fraud','scam','e-commerce','amazon','flipkart','bill','receipt','damaged','quality'],
            'labor': ['salary','wages','fired','terminated','employer','employee','workplace','harassment','leave','notice period','resignation','pf','provident fund','gratuity','bonus','overtime','unfair dismissal','retrench','layoff','transfer','sexual harassment','posh','esi','workman','factory','industrial dispute'],
            'constitutional': ['fundamental right','right to','freedom','equality','discrimination','caste','religion','sc/st','reservation','rti','right to information','pil','public interest','free speech','censorship','protest','arrest','bail','habeas corpus','detention','custodial','encounter','police brutality','privacy right','aadhaar']
        }

    def initial_classify(self, query: str) -> tuple:
        q = query.lower()
        scores = {}
        for case_type, keywords in self.case_keywords.items():
            match_count = sum(1 for kw in keywords if kw in q)
            scores[case_type] = match_count

        if max(scores.values()) == 0:
            return 'general', 0.5

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 5.0, 1.0)
        return best_type, confidence

# -------------------------
# Entity Extraction Agent (Regex-based)
# -------------------------
class EntityExtractionAgent:
    def extract(self, text: str):
        out = {'dates': [], 'locations': [], 'values': [], 'items': [], 'parties': []}

        # Extract dates
        out['dates'] = re.findall(r'\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b', text)[:3]
        months = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?', text, flags=re.I)
        out['dates'] += months
        
        # Extract values
        out['values'] = re.findall(r'\b(?:Rs\.?|₹)\s*[\d,]+\b', text)[:3]
        
        # Extract locations (capitalized words)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', text)
        out['locations'] = caps[:4]
        
        # Extract items
        keywords = ['phone','laptop','car','house','land','jewelry','money','document','agreement','FIR','complaint']
        out['items'] = [kw for kw in keywords if kw.lower() in text.lower()][:4]
        
        return {k:list(dict.fromkeys(v)) for k,v in out.items()}

# -------------------------
# Adaptive Question Generator (FIXED)
# -------------------------
class AdaptiveQuestionGenerator:
    OFFENSE_KEYWORDS = {
        'murder':  ['murder','killed','homicide','stabbed','shot','strangled','burned','ipc 302','302'],
        'theft':   ['theft','stolen','pickpocket','burglary','phone was stolen','ipc 379','379'],
        'robbery': ['robbery','dacoity','snatched','armed','weapon','force','ipc 392','392','395','396'],
        'assault': ['assault','beat','injury','attack','fight','ipc 323','324','325','326'],
    }

    QUESTION_TEMPLATES = {
        # CRIMINAL CASES
        'murder': {
            'time':       ['When did the incident occur?', 'Approximate time of death known?'],
            'location':   ['Where did the incident happen?', 'Where was the body found?'],
            'relationship':['What was the relationship between accused and victim?', 'Any prior dispute?'],
            'weapon':     ['What weapon was used? (knife/firearm/blunt object)', 'Was the weapon recovered?'],
            'evidence':   ['Post-mortem report available?', 'Any CCTV or eyewitnesses?'],
            'police':     ['FIR filed? Which police station?', 'Any arrests made?'],
        },
        'robbery': {
            'time':     ['When did the robbery occur?'],
            'location': ['Where exactly did it happen? (street / shop / home)'],
            'force':    ['Was a weapon or threat used?', 'Any injuries?'],
            'property': ['What items or money were taken?'],
            'evidence': ['Any CCTV or witnesses?', 'Police informed? FIR filed?']
        },
        'theft': {
            'time':     ['When was the item last seen?', 'When did you notice it missing?'],
            'location': ['Where was the theft location?'],
            'property': ['What item was stolen? (model/serial/IMEI)'],
            'evidence': ['CCTV or witnesses?', 'Proof of ownership available?'],
            'police':   ['FIR filed? Which station?']
        },
        'assault': {
            'time':     ['When did the assault occur?'],
            'location': ['Where did it happen?'],
            'injury':   ['What injuries occurred? Medical report available?'],
            'cause':    ['Was there a dispute or trigger?'],
            'evidence': ['CCTV or witnesses?', 'Any hospital or police report?']
        },
        
        # PROPERTY CASES
        'property': {
            'type':     ['What type of property dispute? (land/house/inheritance/tenant)'],
            'location': ['Where is the property located? (address/survey number)'],
            'ownership':['Do you have ownership documents? (sale deed/title deed)'],
            'dispute':  ['What is the exact nature of the dispute?'],
            'timeline': ['When did the dispute start?'],
            'parties':  ['Who are the other parties involved?'],
            'documents':['Do you have: mutation records, property tax receipts, court orders?']
        },
        
        # FAMILY CASES
        'family': {
            'type':     ['What is the family matter? (divorce/custody/maintenance/inheritance)'],
            'marriage': ['When and where did the marriage take place?'],
            'duration': ['How long have you been married/separated?'],
            'children': ['Do you have children? If yes, their ages?'],
            'grounds':  ['What are the grounds for divorce/dispute?'],
            'attempts': ['Have you tried mediation or counseling?'],
            'documents':['Do you have: marriage certificate, proof of income, other relevant documents?']
        },
        
        # CONTRACT CASES
        'contract': {
            'type':     ['What type of agreement? (sale/loan/service/employment)'],
            'date':     ['When was the contract signed?'],
            'amount':   ['What is the contract value/amount involved?'],
            'breach':   ['How has the contract been breached?'],
            'timeline': ['When did the breach occur?'],
            'written':  ['Do you have a written agreement?'],
            'remedy':   ['What remedy are you seeking? (refund/specific performance/damages)']
        },
        
        # CYBER / PRIVACY CASES
        'cyber': {
            'platform': ['On which platform did this happen? (WhatsApp/Instagram/Facebook/Twitter/website/email)'],
            'content':  ['What type of content was leaked or misused? (private chats/photos/videos/personal data)'],
            'person':   ['Do you know who is responsible for this? What is your relationship with them?'],
            'timeline': ['When did you first discover this had happened?'],
            'spread':   ['How widely has the content spread? (sent to specific people / posted publicly / went viral)'],
            'evidence': ['Do you have screenshots or proof of the leak/misuse?'],
            'reported': ['Have you reported this to the platform or filed a cyber crime complaint?']
        },

        # CONSUMER CASES
        'consumer': {
            'product':  ['What product or service is this about?'],
            'seller':   ['Who is the seller or service provider? (company name/website)'],
            'amount':   ['How much did you pay? Do you have the bill or receipt?'],
            'defect':   ['What exactly went wrong? (defective product / service failure / misleading claims)'],
            'timeline': ['When did you purchase it, and when did the problem start?'],
            'attempt':  ['Have you contacted the seller for a refund or replacement? What was their response?'],
            'documents':['Do you have: invoice, warranty card, written complaint, email correspondence?']
        },

        # LABOR / EMPLOYMENT CASES
        'labor': {
            'role':     ['What is/was your job role and how long have you worked there?'],
            'issue':    ['What is the specific workplace issue? (termination/salary dispute/harassment/PF)'],
            'employer': ['What type of employer is this? (private company / government / startup / factory)'],
            'timeline': ['When did this issue begin?'],
            'contract': ['Do you have an employment contract or offer letter?'],
            'action':   ['Have you raised this with HR or any internal committee?'],
            'documents':['Do you have: appointment letter, salary slips, termination notice, email evidence?']
        },

        # CONSTITUTIONAL / RIGHTS CASES
        'constitutional': {
            'right':    ['Which fundamental right do you believe has been violated?'],
            'authority':['Which government body or authority is involved?'],
            'incident': ['Describe the specific incident or action that violated your rights.'],
            'timeline': ['When did this happen?'],
            'impact':   ['How has this affected you personally?'],
            'action':   ['Have you filed any complaint (police/RTI/PIL) about this?'],
            'evidence': ['Do you have evidence? (documents/photos/video/witnesses)']
        }
    }

    # Topic-specific follow-up question patterns for truly adaptive questions.
    # When the user's query mentions a keyword, these targeted follow-ups are used
    # instead of generic template questions.
    TOPIC_FOLLOWUPS = {
        'leak':     ['Who leaked the content? Do you know them personally?', 'Was the content shared without your consent?', 'Has the leaked content caused you reputational or emotional harm?'],
        'private chat': ['Were these private conversations between two people, or a group?', 'On which messaging platform were the chats originally sent?', 'Were the chats altered or taken out of context before being shared?'],
        'screenshot': ['Were the screenshots taken with your knowledge?', 'Have the screenshots been shared publicly or with specific people?'],
        'photo':    ['Were these photos taken with your consent?', 'Were the photos intimate or personal in nature?', 'Have these photos been shared or uploaded online?'],
        'video':    ['Was this video recorded with your knowledge and consent?', 'Has the video been uploaded to any website or social media?', 'Is the video being used to threaten or blackmail you?'],
        'hack':     ['What type of account was compromised? (email/social media/bank)', 'Have you noticed unauthorized transactions or messages?', 'Have you changed your passwords and enabled 2FA since the breach?'],
        'defam':    ['What defamatory statements were made about you?', 'Was the defamation published online or in traditional media?', 'Can you prove that the statements are false?'],
        'stalk':    ['How long has the stalking been going on?', 'Is the stalker known to you?', 'Do you have records of the stalking behavior? (messages/call logs/location tracking)'],
        'blackmail':['What is the person threatening to do?', 'Have you paid any money to the blackmailer so far?', 'Do you have proof of the threats? (messages/recordings)'],
        'lost':     ['Where and when was the item lost?', 'Did you try to contact the person who found it?', 'Do you have proof of ownership? (purchase receipt/IMEI/serial number)'],
        'found':    ['Where and when was the item found?', 'Has anyone claimed the item?', 'Did you report the found item to local police?'],
        'refund':   ['How much is the refund amount?', 'How long ago did you request the refund?', 'What reason did the seller give for refusing the refund?'],
        'fired':    ['Were you given any reason for the termination?', 'Did you receive a termination letter?', 'Were you serving a notice period?'],
        'salary':   ['How many months of salary are pending?', 'Have you raised this with HR or management in writing?', 'Do you have salary slips or bank statements as proof?'],
        'harassment':['What type of harassment? (sexual/verbal/workplace bullying)', 'Have you reported this to an internal committee?', 'Do you have witnesses or evidence of the harassment?'],
        'arrest':   ['Who was arrested and when?', 'What charges have been filed?', 'Has a lawyer been contacted? Has bail been applied for?'],
        'bail':     ['What is the offense the person is charged with?', 'How long have they been in custody?', 'Is this a bailable or non-bailable offense?'],
        'discrimination': ['On what basis were you discriminated against? (caste/religion/gender/disability)', 'Where did this discrimination occur? (workplace/public place/institution)', 'Do you have evidence of the discriminatory treatment?'],
        'divorce':  ['How long have you been married?', 'What are the grounds for seeking divorce? (cruelty/adultery/desertion/mutual consent)', 'Are there children involved? If yes, their ages?'],
        'custody':  ['How old are the children?', 'Who currently has physical custody?', 'Has either parent been abusive or neglectful?'],
        'rent':     ['Is there a written rental agreement?', 'What is the monthly rent and how long have you been a tenant?', 'What is the dispute about? (eviction/deposit/maintenance)'],
        'eviction': ['Do you have a written lease/rental agreement?', 'On what grounds is the eviction being sought?', 'How long have you been living at this property?'],
    }

    def detect_crime_subtype(self, text: str) -> str:
        t = text.lower()
        best = 'theft'
        best_score = 0
        for subtype, keywords in self.OFFENSE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in t)
            if score > best_score:
                best = subtype
                best_score = score
        return best

    def _find_topic_questions(self, situation: str, asked: List[str]) -> Optional[str]:
        """Scan the user's query for topic keywords and return a contextual follow-up question."""
        s = situation.lower()
        for topic, questions in self.TOPIC_FOLLOWUPS.items():
            if topic in s:
                for q in questions:
                    if q not in asked:
                        logger.info(f"Topic-based question (topic='{topic}'): {q}")
                        return q
        return None

    def generate_next(self, case_type: str, kg: KnowledgeGraph, asked: List[str], history: List[str] = None) -> Optional[str]:
        """Generate next question based on case type and history, ensuring no repeats.
        Uses a priority chain: Ollama LLM -> topic-based contextual -> category templates."""
        situation = kg.context.get('situation', '')
        history = history or []

        # --- Priority 1: Use Ollama LLM if available ---
        if ollama_available:
            try:
                import httpx
                facts = "\n".join([f"{k}: {', '.join(v)}" for k, v in kg.entities.items() if v])
                prompt = (
                    f"You are a smart legal assistant gathering facts. Analyze the scenario and ask ONE short, relevant follow-up question to get more details.\n\n"
                    f"Case: {case_type}\n"
                    f"Situation: {situation}\n"
                    f"Facts collected: {facts}\n"
                    f"Conversation history: {history}\n\n"
                    f"Output ONLY the exact question text. Keep it under 15 words. DO NOT ask anything already answered in the history."
                )
                resp = httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 40}}, timeout=10)
                if resp.status_code == 200:
                    q = resp.json().get("response", "").strip().strip('"').strip("'")
                    if q and not q.startswith("Here is") and not q.startswith("Question"):
                        logger.info(f"Ollama contextual question: {q}")
                        return q
            except Exception as e:
                logger.warning(f"Ollama question generation failed: {e}")

        # --- Priority 2: Topic-based contextual questions from the user's own words ---
        # Combine the original situation with all answers so far for richer topic matching
        combined_text = situation + ' ' + ' '.join(history)
        topic_q = self._find_topic_questions(combined_text, asked)
        if topic_q:
            return topic_q

        # --- Priority 3: Category-specific template questions ---
        if case_type == 'criminal':
            subtype = kg.context.get('criminal_subtype') or self.detect_crime_subtype(situation)
            templates = self.QUESTION_TEMPLATES.get(subtype, self.QUESTION_TEMPLATES['theft'])
        else:
            templates = self.QUESTION_TEMPLATES.get(case_type, self.QUESTION_TEMPLATES.get('general', {}))

        for category, questions in templates.items():
            missing_questions = [q for q in questions if q not in asked]
            if missing_questions:
                logger.info(f"Next question from category '{category}': {missing_questions[0]}")
                return missing_questions[0]

        logger.info("All questions exhausted")
        return None

# -------------------------
# AI-Powered Analyzer — Ollama / llama3.2
# -------------------------
class AIAnalyzer:
    def __init__(self):
        self.has_llm = ollama_available

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def _build_prompt(self, case_type: str, kg: KnowledgeGraph) -> str:
        situation = kg.context.get('situation', '')
        subtype = kg.context.get('criminal_subtype', '') if case_type == 'criminal' else case_type

        facts_parts = []
        for k, v in kg.entities.items():
            if v:
                facts_parts.append(f"{k}: {', '.join(v)}")
        facts_str = '\n'.join(facts_parts) if facts_parts else 'None collected yet'

        return f"""This is a law school exam question about Indian criminal and civil law. Answer comprehensively as a law professor would.

SCENARIO ({case_type.upper()} - {subtype}):
{situation}

ADDITIONAL FACTS:
{facts_str}

For this scenario, explain in detail:
1. Which sections of the Indian Penal Code (IPC) or Civil Procedure Code (CPC) apply and why
2. The standard procedure a complainant should follow step-by-step
3. Types of evidence that would strengthen the case
4. The legal rights available to the person in this scenario under Indian law
5. The typical legal timeline and what happens at each stage

Be specific with IPC section numbers. Write as a detailed academic answer."""

    # ------------------------------------------------------------------
    # Call Ollama REST API (non-streaming)
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> Optional[str]:
        if not ollama_available:
            return None
        try:
            import httpx
            logger.info(f"🦙 Calling Ollama ({OLLAMA_MODEL})...")
            resp = httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 600,
                    }
                },
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze(self, case_type: str, kg: KnowledgeGraph) -> str:
        situation = kg.context.get('situation', '')
        subtype = kg.context.get('criminal_subtype', 'Unknown') if case_type == 'criminal' else case_type

        fact_text = "\n".join([
            f"{k.upper()}: {', '.join(v)}"
            for k, v in kg.entities.items() if v
        ])

        # --- Try Ollama / llama3.2 first ---
        if ollama_available:
            prompt = self._build_prompt(case_type, kg)
            llm_response = self._call_ollama(prompt)
            if llm_response:
                return self._format_llm_report(case_type, subtype, situation, fact_text, llm_response)

        # --- Fallback: template-based analysis ---
        logger.info("Ollama unavailable — using template analysis")
        return self._generate_enhanced_report(case_type, subtype, situation, fact_text, kg)

    # ------------------------------------------------------------------
    # Format LLM output as a structured report
    # ------------------------------------------------------------------
    def _format_llm_report(self, case_type, subtype, situation, facts, llm_response):
        return f"""═══════════════════════════════════════════════════
LEGAL CONSULTATION REPORT  [AI Generated — llama3.2]
═══════════════════════════════════════════════════

Case Type: {case_type.upper()} — {subtype}
Model: llama3.2 via Ollama (local)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────
CLIENT STATEMENT:
{situation}

─────────────────────────────────────────────────
FACTS EXTRACTED:
{facts if facts else 'Limited information — answer all questions for a more detailed analysis'}

─────────────────────────────────────────────────
AI LEGAL ANALYSIS (llama3.2):

{llm_response}

─────────────────────────────────────────────────
DISCLAIMER:
This is an AI-generated preliminary analysis. Please consult a
qualified legal professional for authoritative case-specific advice.
═══════════════════════════════════════════════════"""
    
    def _format_report(self, case_type, subtype, situation, facts, analysis):
        return f"""═══════════════════════════════════════════════════
LEGAL CONSULTATION REPORT
═══════════════════════════════════════════════════

Case Type: {case_type.upper()}
Subtype: {subtype}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────
CLIENT STATEMENT:
{situation}

─────────────────────────────────────────────────
FACTS EXTRACTED:
{facts if facts else 'Limited information available'}

─────────────────────────────────────────────────
AI LEGAL ANALYSIS:
{analysis}

─────────────────────────────────────────────────
DISCLAIMER:
This is an AI-generated preliminary analysis. Please consult
a qualified legal professional for authoritative advice.
═══════════════════════════════════════════════════"""
    
    def _generate_enhanced_report(self, case_type, subtype, situation, facts, kg: KnowledgeGraph):
        """Generate intelligent analysis based on case facts"""
        
        # Get collected information
        dates = kg.entities.get('dates', [])
        locations = kg.entities.get('locations', [])
        values = kg.entities.get('values', [])
        items = kg.entities.get('items', [])
        
        # Analyze based on case type and facts
        if case_type == 'criminal' and subtype == 'robbery':
            return self._analyze_robbery_case(situation, dates, locations, values, items, facts)
        elif case_type == 'criminal' and subtype == 'theft':
            return self._analyze_theft_case(situation, dates, locations, values, items, facts)
        elif case_type == 'criminal' and subtype == 'murder':
            return self._analyze_murder_case(situation, dates, locations, facts)
        elif case_type == 'criminal' and subtype == 'assault':
            return self._analyze_assault_case(situation, dates, locations, facts)
        elif case_type == 'property':
            return self._analyze_property_case(situation, dates, locations, values, facts)
        elif case_type == 'family':
            return self._analyze_family_case(situation, dates, facts)
        elif case_type == 'contract':
            return self._analyze_contract_case(situation, dates, values, facts)
        elif case_type == 'cyber':
            return self._analyze_cyber_case(situation, dates, facts, kg)
        elif case_type == 'consumer':
            return self._analyze_consumer_case(situation, dates, values, facts)
        elif case_type == 'labor':
            return self._analyze_labor_case(situation, dates, facts)
        elif case_type == 'constitutional':
            return self._analyze_constitutional_case(situation, dates, facts)
        else:
            return self._generate_smart_general_report(case_type, subtype, situation, facts, kg)
    
    def _analyze_robbery_case(self, situation, dates, locations, values, items, facts):
        """Detailed robbery case analysis"""
        
        # Build context-aware analysis
        analysis = []
        
        analysis.append("CASE OVERVIEW:")
        analysis.append(f"This is a robbery case under IPC Section 390-392 (Robbery and Dacoity).")
        
        if dates:
            analysis.append(f"The incident occurred on {dates[0]}. Time is crucial - report immediately.")
        
        if locations:
            analysis.append(f"Location: {locations[0]}. Evidence collection at the crime scene is vital.")
        
        if values:
            analysis.append(f"Value involved: {values[0]}. Higher amounts may lead to more severe charges.")
        
        if items:
            analysis.append(f"Items taken: {', '.join(items)}. Document with purchase receipts/serial numbers.")
        
        analysis.append("\nAPPLICABLE LAWS:")
        analysis.append("• IPC Section 390: Definition of Robbery (theft with force/threat)")
        analysis.append("• IPC Section 392: Punishment for robbery (up to 10 years + fine)")
        analysis.append("• If weapon used: IPC Section 397 (robbery with deadly weapon) - up to 14 years")
        analysis.append("• If injury caused: Enhanced punishment under relevant sections")
        
        analysis.append("\nIMMEDIATE ACTIONS REQUIRED:")
        analysis.append("1. File FIR immediately at the nearest police station (jurisdiction based on crime location)")
        analysis.append("2. Provide detailed description of perpetrators if seen")
        analysis.append("3. Request police to preserve CCTV footage from the area")
        analysis.append("4. Get medical examination done if any injuries sustained")
        analysis.append("5. Prepare list of stolen items with proof of ownership")
        analysis.append("6. Identify and contact witnesses immediately")
        
        analysis.append("\nEVIDENCE TO COLLECT:")
        analysis.append("• CCTV footage from crime scene and surrounding areas")
        analysis.append("• Witness statements (get written statements if possible)")
        analysis.append("• Photos of crime scene and any damage")
        analysis.append("• Medical reports if injuries present")
        analysis.append("• Purchase receipts/serial numbers of stolen items")
        analysis.append("• Bank statements showing cash withdrawal (if cash stolen)")
        
        analysis.append("\nLEGAL TIMELINE:")
        analysis.append("• FIR should be filed within 24 hours for best results")
        analysis.append("• CCTV footage may be overwritten after 7-30 days")
        analysis.append("• Witness memory fades - record statements quickly")
        
        analysis.append("\nNEXT STEPS:")
        analysis.append("1. File FIR today if not already done")
        analysis.append("2. Engage a criminal lawyer to follow up on investigation")
        analysis.append("3. Monitor police investigation progress")
        analysis.append("4. Be prepared to identify accused if caught")
        analysis.append("5. Keep all evidence organized for trial")
        
        return f"""═══════════════════════════════════════════════════
LEGAL CONSULTATION REPORT - ROBBERY CASE
═══════════════════════════════════════════════════

Case Type: CRIMINAL - ROBBERY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────
CLIENT STATEMENT:
{situation}

─────────────────────────────────────────────────
FACTS COLLECTED:
{facts if facts else 'Limited information - answer all questions for detailed analysis'}

─────────────────────────────────────────────────
LEGAL ANALYSIS:

{chr(10).join(analysis)}

─────────────────────────────────────────────────
CRITICAL REMINDERS:
⚠ Time is of the essence - evidence deteriorates quickly
⚠ FIR must be filed immediately
⚠ This is a serious offense - professional legal representation recommended
⚠ Cooperate fully with police investigation

─────────────────────────────────────────────────
DISCLAIMER:
This is a preliminary legal analysis based on the information provided.
For case-specific advice and representation, please consult a qualified
criminal lawyer immediately. Laws and procedures may vary by state.
═══════════════════════════════════════════════════"""
    
    def _analyze_theft_case(self, situation, dates, locations, values, items, facts):
        """Similar detailed analysis for theft"""
        return self._generate_template_report('criminal', 'theft', situation, facts)
    
    def _analyze_murder_case(self, situation, dates, locations, facts):
        """Similar detailed analysis for murder"""
        return self._generate_template_report('criminal', 'murder', situation, facts)
    
    def _analyze_assault_case(self, situation, dates, locations, facts):
        """Similar detailed analysis for assault"""
        return self._generate_template_report('criminal', 'assault', situation, facts)
    
    def _analyze_property_case(self, situation, dates, locations, values, facts):
        """Detailed property dispute analysis"""
        
        analysis = []
        
        analysis.append("CASE OVERVIEW:")
        analysis.append("This is a property dispute case under relevant civil/property laws.")
        
        if locations:
            analysis.append(f"Property location: {locations[0]}. Survey records and mutation documents are crucial.")
        
        if values:
            analysis.append(f"Estimated value: {values[0]}. Property valuation report recommended.")
        
        analysis.append("\nAPPLICABLE LAWS:")
        analysis.append("• Transfer of Property Act, 1882")
        analysis.append("• Indian Succession Act, 1925 (if inheritance dispute)")
        analysis.append("• Specific Relief Act, 1963 (for specific performance)")
        analysis.append("• Registration Act, 1908 (for property registration)")
        analysis.append("• State-specific Land Revenue Acts")
        
        analysis.append("\nIMMEDIATE ACTIONS REQUIRED:")
        analysis.append("1. Collect all property documents (sale deed, title deed, mutation records)")
        analysis.append("2. Get property survey done to verify boundaries")
        analysis.append("3. Check encumbrance certificate from sub-registrar office")
        analysis.append("4. Verify ownership chain - trace back 30 years minimum")
        analysis.append("5. Check for any pending litigation on the property")
        analysis.append("6. Document any illegal occupation or encroachment with photos/videos")
        
        analysis.append("\nDOCUMENTS TO COLLECT:")
        analysis.append("• Sale/Purchase deed")
        analysis.append("• Title deed and ownership chain")
        analysis.append("• Mutation records (7/12 extract, khasra, etc.)")
        analysis.append("• Property tax receipts")
        analysis.append("• Encumbrance certificate")
        analysis.append("• Survey/plot plan")
        analysis.append("• Building plan approval (if applicable)")
        analysis.append("• Will/succession certificate (if inheritance case)")
        
        analysis.append("\nRESOLUTION OPTIONS:")
        analysis.append("1. Negotiation and settlement (fastest and cheapest)")
        analysis.append("2. Mediation through court or private mediator")
        analysis.append("3. Civil suit in appropriate court")
        analysis.append("4. Partition suit (if co-owned property)")
        analysis.append("5. Injunction to prevent alienation/damage")
        
        analysis.append("\nNEXT STEPS:")
        analysis.append("1. Consult a property lawyer with all documents")
        analysis.append("2. Get legal opinion on ownership status")
        analysis.append("3. Attempt amicable settlement first")
        analysis.append("4. If settlement fails, file appropriate civil suit")
        analysis.append("5. Apply for interim injunction if urgent")
        
        return f"""═══════════════════════════════════════════════════
LEGAL CONSULTATION REPORT - PROPERTY DISPUTE
═══════════════════════════════════════════════════

Case Type: PROPERTY DISPUTE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────
CLIENT STATEMENT:
{situation}

─────────────────────────────────────────────────
FACTS COLLECTED:
{facts if facts else 'Limited information - answer all questions for detailed analysis'}

─────────────────────────────────────────────────
LEGAL ANALYSIS:

{chr(10).join(analysis)}

─────────────────────────────────────────────────
CRITICAL REMINDERS:
⚠ Property disputes can take years - document everything
⚠ Verify all documents before making any payment
⚠ Get title search done by professional lawyer
⚠ Do not make any physical changes to disputed property

─────────────────────────────────────────────────
DISCLAIMER:
This is a preliminary legal analysis based on the information provided.
Property laws vary by state. Please consult a qualified property lawyer
for case-specific advice and representation.
═══════════════════════════════════════════════════"""
    
    def _analyze_family_case(self, situation, dates, facts):
        """Detailed family law analysis"""
        return self._generate_template_report('family', 'family', situation, facts)
    
    def _analyze_contract_case(self, situation, dates, values, facts):
        """Detailed contract dispute analysis"""
        return self._generate_template_report('contract', 'contract', situation, facts)
    
    def _analyze_cyber_case(self, situation, dates, facts, kg):
        """Detailed cyber/privacy law analysis"""
        s = situation.lower()
        analysis = []
        
        analysis.append("CASE OVERVIEW:")
        if 'leak' in s or 'share' in s or 'forward' in s or 'private chat' in s or 'chat' in s:
            analysis.append("This involves unauthorized sharing of private communications, which is punishable under Indian cyber and privacy laws.")
        elif 'hack' in s:
            analysis.append("This involves unauthorized access to computer systems or accounts, a serious offense under the IT Act.")
        elif 'stalk' in s or 'bully' in s:
            analysis.append("This involves cyber stalking or bullying, which is a recognized criminal offense under Indian law.")
        else:
            analysis.append("This is a cyber/digital privacy matter that falls under the Information Technology Act, 2000.")
        
        if dates:
            analysis.append(f"Timeline: {dates[0]}. Digital evidence should be preserved immediately.")

        analysis.append("\nAPPLICABLE LAWS:")
        analysis.append("  IT Act, 2000 (Information Technology Act):")
        analysis.append("  - Section 43: Unauthorized access - compensation up to Rs. 1 crore")
        analysis.append("  - Section 66: Computer related offenses - up to 3 years + fine")
        analysis.append("  - Section 66C: Identity theft - up to 3 years + Rs. 1 lakh fine")
        analysis.append("  - Section 66D: Cheating by impersonation using computer - up to 3 years + fine")
        analysis.append("  - Section 66E: Violation of privacy (capturing/publishing private images) - up to 3 years + Rs. 2 lakh fine")
        analysis.append("  - Section 67: Publishing obscene material electronically - up to 5 years + Rs. 10 lakh fine")
        analysis.append("  - Section 67A: Publishing sexually explicit material - up to 7 years + Rs. 10 lakh fine")
        analysis.append("  - Section 72: Breach of confidentiality and privacy - up to 2 years + Rs. 1 lakh fine")
        analysis.append("")
        analysis.append("  Indian Penal Code (IPC):")
        analysis.append("  - Section 499/500: Criminal defamation - up to 2 years + fine")
        analysis.append("  - Section 354C: Voyeurism - up to 3 years (first offense)")
        analysis.append("  - Section 354D: Stalking (including cyber stalking) - up to 3 years")
        analysis.append("  - Section 506: Criminal intimidation/blackmail - up to 2-7 years")
        analysis.append("  - Section 509: Word/gesture to insult modesty - up to 3 years")
        
        analysis.append("\nIMMEDIATE ACTIONS:")
        analysis.append("1. Take screenshots of ALL leaked/shared content as evidence (with timestamps and URLs)")
        analysis.append("2. Report to the platform (WhatsApp/Instagram/Facebook) to get the content taken down")
        analysis.append("3. File a complaint on the National Cyber Crime Portal: https://cybercrime.gov.in")
        analysis.append("4. Visit your nearest Cyber Crime Police Station to file an FIR")
        analysis.append("5. Do NOT delete your own copies of the original conversations - they are evidence")
        analysis.append("6. Do NOT engage with or confront the perpetrator online")

        analysis.append("\nEVIDENCE TO PRESERVE:")
        analysis.append("  - Screenshots with timestamps and URLs")
        analysis.append("  - Original chat/message history (do NOT delete)")
        analysis.append("  - Metadata (phone numbers, email addresses, usernames involved)")
        analysis.append("  - Witness accounts (anyone who saw the shared content)")
        analysis.append("  - Any threatening/blackmail messages received")
        
        analysis.append("\nLEGAL REMEDIES AVAILABLE:")
        analysis.append("1. Criminal complaint under IT Act and IPC")
        analysis.append("2. Civil suit for damages (compensation for emotional distress)")
        analysis.append("3. Injunction order from court to remove content")
        analysis.append("4. Right to be Forgotten (under data protection principles)")
        analysis.append("5. Intermediary liability - platform can be held liable for not removing content after notice")

        return f"""=======================================================
LEGAL CONSULTATION REPORT - CYBER / PRIVACY CASE
=======================================================

Case Type: CYBER / PRIVACY LAW
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS COLLECTED:
{facts if facts else 'Limited information - answer all questions for detailed analysis'}

-------------------------------------------------------
LEGAL ANALYSIS:

{chr(10).join(analysis)}

-------------------------------------------------------
CRITICAL REMINDERS:
! Digital evidence can be deleted by perpetrators - preserve it NOW
! File cyber crime complaint within 72 hours for best results
! Cyber Crime Helpline: 1930 (toll-free)
! National Cyber Crime Portal: https://cybercrime.gov.in
! Do NOT retaliate online - it may weaken your legal position

-------------------------------------------------------
DISCLAIMER:
This is a preliminary legal analysis. Cyber laws evolve rapidly.
Please consult a lawyer specializing in cyber law for case-specific advice.
======================================================="""

    def _analyze_consumer_case(self, situation, dates, values, facts):
        """Detailed consumer rights analysis"""
        analysis = []
        analysis.append("CASE OVERVIEW:")
        analysis.append("This is a consumer rights matter under the Consumer Protection Act, 2019.")
        if values:
            analysis.append(f"Amount involved: {values[0]}.")
        
        analysis.append("\nAPPLICABLE LAWS:")
        analysis.append("  - Consumer Protection Act, 2019 (replaced the 1986 Act)")
        analysis.append("  - E-Commerce Rules, 2020 (for online purchases)")
        analysis.append("  - Bureau of Indian Standards Act (product quality)")
        analysis.append("  - Legal Metrology Act (weights and measures)")
        analysis.append("  - FSSAI regulations (if food related)")
        
        analysis.append("\nCONSUMER FORUM JURISDICTION:")
        analysis.append("  - Up to Rs. 1 crore: District Consumer Forum")
        analysis.append("  - Rs. 1 crore to Rs. 10 crore: State Consumer Commission")
        analysis.append("  - Above Rs. 10 crore: National Consumer Commission")
        
        analysis.append("\nIMMEDIATE STEPS:")
        analysis.append("1. Send a formal written complaint to the seller/service provider")
        analysis.append("2. Keep copies of all bills, warranties, and communications")
        analysis.append("3. File complaint on National Consumer Helpline: 1800-11-4000")
        analysis.append("4. File online complaint at https://consumerhelpline.gov.in")
        analysis.append("5. If unresolved, file case at District Consumer Forum (minimal fees)")
        
        analysis.append("\nCONSUMER RIGHTS:")
        analysis.append("  - Right to safety, information, choice, redressal")
        analysis.append("  - Right to refund/replacement for defective products")
        analysis.append("  - Right to compensation for deficiency in service")
        analysis.append("  - Protection against unfair trade practices")

        return f"""=======================================================
LEGAL CONSULTATION REPORT - CONSUMER RIGHTS
=======================================================

Case Type: CONSUMER PROTECTION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS COLLECTED:
{facts if facts else 'Limited information provided'}

-------------------------------------------------------
LEGAL ANALYSIS:

{chr(10).join(analysis)}

-------------------------------------------------------
HELPLINE:
! National Consumer Helpline: 1800-11-4000 (toll-free)
! Online: https://consumerhelpline.gov.in
! Consumer forum cases have NO court fee for claims under Rs. 5 lakh

-------------------------------------------------------
DISCLAIMER:
This is a preliminary analysis. Consumer protection laws provide
strong remedies. Consult a consumer rights lawyer for detailed guidance.
======================================================="""

    def _analyze_labor_case(self, situation, dates, facts):
        """Detailed labor/employment law analysis"""
        s = situation.lower()
        analysis = []
        analysis.append("CASE OVERVIEW:")
        analysis.append("This is an employment/labor law matter.")
        
        analysis.append("\nAPPLICABLE LAWS:")
        if 'harass' in s or 'posh' in s or 'sexual' in s:
            analysis.append("  - Sexual Harassment of Women at Workplace Act, 2013 (POSH Act)")
            analysis.append("  - Every organization with 10+ employees must have an Internal Complaints Committee (ICC)")
        if 'fired' in s or 'terminat' in s or 'dismiss' in s:
            analysis.append("  - Industrial Disputes Act, 1947 (unfair dismissal)")
            analysis.append("  - Shops and Establishments Act (applicable state)")
        analysis.append("  - Payment of Wages Act, 1936")
        analysis.append("  - Payment of Gratuity Act, 1972")
        analysis.append("  - Employees Provident Fund Act, 1952")
        analysis.append("  - Industrial Employment (Standing Orders) Act")
        
        analysis.append("\nIMMEDIATE STEPS:")
        analysis.append("1. Document everything in writing (emails, letters, notices)")
        analysis.append("2. Keep copies of employment contract, salary slips, offer letter")
        analysis.append("3. File complaint with Labour Commissioner if wages are due")
        analysis.append("4. If harassment: report to Internal Complaints Committee (ICC)")
        analysis.append("5. Consult a labor lawyer before signing any settlement")
        
        analysis.append("\nEMPLOYEE RIGHTS:")
        analysis.append("  - Right to timely payment of wages")
        analysis.append("  - Right to PF, gratuity, and bonus as applicable")
        analysis.append("  - Protection against unfair dismissal")
        analysis.append("  - Right to safe and harassment-free workplace")
        analysis.append("  - Right to notice period or pay in lieu")

        return f"""=======================================================
LEGAL CONSULTATION REPORT - EMPLOYMENT / LABOR
=======================================================

Case Type: LABOR / EMPLOYMENT LAW
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS COLLECTED:
{facts if facts else 'Limited information provided'}

-------------------------------------------------------
LEGAL ANALYSIS:

{chr(10).join(analysis)}

-------------------------------------------------------
HELPLINE:
! Labour Commissioner office in your district
! Shram Suvidha Portal: https://shramsuvidha.gov.in
! EPFO Helpline: 1800-118-005

-------------------------------------------------------
DISCLAIMER:
This is a preliminary analysis. Employment law varies by state.
Consult a labor law specialist for case-specific advice.
======================================================="""

    def _analyze_constitutional_case(self, situation, dates, facts):
        """Constitutional/fundamental rights analysis"""
        analysis = []
        analysis.append("CASE OVERVIEW:")
        analysis.append("This matter involves constitutional and fundamental rights guaranteed under the Indian Constitution.")
        
        analysis.append("\nFUNDAMENTAL RIGHTS (Part III, Constitution):")
        analysis.append("  - Article 14: Right to Equality")
        analysis.append("  - Article 19: Freedom of speech, assembly, movement, profession")
        analysis.append("  - Article 21: Right to Life and Personal Liberty (includes right to privacy)")
        analysis.append("  - Article 22: Protection against arbitrary arrest and detention")
        analysis.append("  - Article 25: Freedom of religion")
        analysis.append("  - Article 32: Right to Constitutional Remedies (Supreme Court)")
        analysis.append("  - Article 226: Right to approach High Court for writ petitions")
        
        analysis.append("\nLEGAL REMEDIES AVAILABLE:")
        analysis.append("1. File a writ petition in High Court (Article 226)")
        analysis.append("2. File a writ petition in Supreme Court (Article 32)")
        analysis.append("3. File a Public Interest Litigation (PIL)")
        analysis.append("4. File RTI to get information from government bodies")
        analysis.append("5. Approach National/State Human Rights Commission")
        analysis.append("6. Approach relevant statutory commissions (SC/ST, Women, Minorities)")

        return f"""=======================================================
LEGAL CONSULTATION REPORT - CONSTITUTIONAL RIGHTS
=======================================================

Case Type: CONSTITUTIONAL / FUNDAMENTAL RIGHTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS COLLECTED:
{facts if facts else 'Limited information provided'}

-------------------------------------------------------
LEGAL ANALYSIS:

{chr(10).join(analysis)}

-------------------------------------------------------
IMPORTANT:
! Fundamental rights are enforceable against the State
! Supreme Court (Art 32) and High Court (Art 226) have writ jurisdiction
! NHRC Helpline: 14433

-------------------------------------------------------
DISCLAIMER:
This is a preliminary analysis. Constitutional matters are complex.
Please consult a senior advocate specializing in constitutional law.
======================================================="""

    def _generate_smart_general_report(self, case_type, subtype, situation, facts, kg):
        """Smart general report that adapts to the situation instead of being generic."""
        s = situation.lower()
        
        # Try to detect what the situation is really about and give relevant advice
        topic_sections = []
        
        if any(w in s for w in ['lost', 'found', 'keep', 'return', 'finder']):
            topic_sections.append("RELEVANT LEGAL PROVISIONS:")
            topic_sections.append("  - IPC Section 403: Dishonest misappropriation of property")
            topic_sections.append("    Keeping a found item without attempting to return it to the owner")
            topic_sections.append("    can be treated as a criminal offense - up to 2 years + fine.")
            topic_sections.append("  - IPC Section 404: If the finder knows the owner, punishment is stricter.")
            topic_sections.append("  - Indian Contract Act S.71: Finder of goods has duty to return to owner.")
            topic_sections.append("")
            topic_sections.append("KEY POINTS:")
            topic_sections.append("  - A finder is legally obligated to make reasonable efforts to find the owner")
            topic_sections.append("  - Keeping found property IS legally considered dishonest misappropriation")
            topic_sections.append("  - File a police complaint for lost property immediately")
            topic_sections.append("  - The finder cannot claim ownership just by possession")
        
        if any(w in s for w in ['noise', 'neighbor', 'nuisance', 'disturb']):
            topic_sections.append("RELEVANT LEGAL PROVISIONS:")
            topic_sections.append("  - IPC Section 268: Public nuisance")
            topic_sections.append("  - Noise Pollution Rules, 2000")
            topic_sections.append("  - CPC - civil suit for injunction against nuisance")
        
        if any(w in s for w in ['accident', 'motor', 'vehicle', 'crash', 'hit']):
            topic_sections.append("RELEVANT LEGAL PROVISIONS:")
            topic_sections.append("  - Motor Vehicles Act, 2019")
            topic_sections.append("  - IPC Section 279: Rash driving")
            topic_sections.append("  - IPC Section 304A: Death by negligence")
            topic_sections.append("  - Motor Accident Claims Tribunal (MACT) for compensation")
        
        if any(w in s for w in ['cheat', 'fraud', 'scam', 'trick', 'deceive', 'fake']):
            topic_sections.append("RELEVANT LEGAL PROVISIONS:")
            topic_sections.append("  - IPC Section 415/420: Cheating and fraud - up to 7 years + fine")
            topic_sections.append("  - IPC Section 406: Criminal breach of trust")
            topic_sections.append("  - File FIR immediately with all evidence")
        
        if not topic_sections:
            topic_sections.append("LEGAL ANALYSIS:")
            topic_sections.append(f"Based on the scenario described, this matter requires professional legal consultation.")
            topic_sections.append(f"")
            topic_sections.append(f"Your situation: {situation[:300]}")
            topic_sections.append(f"")
            topic_sections.append("RECOMMENDED STEPS:")
            topic_sections.append("1. Consult a lawyer who specializes in this area of law")
            topic_sections.append("2. Gather all relevant documents, communications, and evidence")
            topic_sections.append("3. Document a detailed timeline of events")
            topic_sections.append("4. Identify and note down details of all parties involved")
            topic_sections.append("5. Consider sending a legal notice before filing a case")
        
        return f"""=======================================================
LEGAL CONSULTATION REPORT
=======================================================

Case Type: {case_type.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS COLLECTED:
{facts if facts else 'Limited information provided'}

-------------------------------------------------------
{chr(10).join(topic_sections)}

-------------------------------------------------------
IMPORTANT NOTES:
  - Act promptly as legal timelines are strict
  - Document everything thoroughly
  - Preserve all evidence (digital and physical)
  - Consult a qualified lawyer for case-specific advice

-------------------------------------------------------
DISCLAIMER:
This is a preliminary analysis based on the information provided.
Please consult a qualified legal professional for authoritative advice.
======================================================="""

    def _generate_template_report(self, case_type, subtype, situation, facts):
        # Fallback template when LLM unavailable
        if case_type == 'criminal':
            steps = """1. File FIR immediately at the nearest police station
2. Collect and preserve all evidence (photos, documents, CCTV)
3. Get witness statements recorded
4. Obtain medical examination report if injuries present
5. Consult a criminal lawyer for detailed legal strategy"""
        elif case_type == 'family':
            steps = """1. Attempt mediation/counseling first if applicable
2. Gather all relevant documents (marriage certificate, financial records)
3. Document any incidents with dates and evidence
4. Consult a family law specialist
5. Consider filing petition in family court if mediation fails"""
        else:
            steps = """1. Gather all relevant documents and evidence
2. Document timeline of events
3. Identify witnesses if any
4. Consult appropriate legal specialist
5. File case in appropriate court if required"""
        
        return f"""=======================================================
LEGAL CONSULTATION REPORT
=======================================================

Case Type: {case_type.upper()}
Subtype: {subtype}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

-------------------------------------------------------
CLIENT STATEMENT:
{situation}

-------------------------------------------------------
FACTS EXTRACTED:
{facts if facts else 'Limited information available'}

-------------------------------------------------------
LEGAL ANALYSIS:

Based on the information provided, this appears to be a {case_type} 
matter requiring immediate attention.

RECOMMENDED LEGAL STEPS:
{steps}

IMPORTANT NOTES:
  - Act promptly as legal timelines are strict
  - Document everything thoroughly
  - Preserve all evidence
  - Consult a qualified lawyer immediately for case-specific advice

-------------------------------------------------------
DISCLAIMER:
This is a preliminary analysis based on limited information.
Please consult a qualified legal professional for authoritative advice.
======================================================="""

# -------------------------
# Agentic Legal System with AI Models
# -------------------------
class AgenticLegalSystem:
    def __init__(self):
        self.classifier = SmartClassifierAgent()
        self.extractor = EntityExtractionAgent()
        self.qgen = AdaptiveQuestionGenerator()
        self.analyzer = AIAnalyzer()
        self.sessions = {}
        self.lock = Lock()

    def start_session(self, query: str, max_turns: int = 7) -> Dict:
        session_id = str(uuid.uuid4())
        case_type, conf = self.classifier.initial_classify(query)
        
        logger.info(f"New session {session_id}: {case_type} (confidence: {conf:.2f})")
        
        kg = KnowledgeGraph()
        kg.set_context('situation', query)
        kg.set_context('case_type', case_type)
        kg.set_context('criminal_subtype', self.qgen.detect_crime_subtype(query))

        # Extract initial entities
        extracted = self.extractor.extract(query)
        for k, vals in extracted.items():
            for v in vals:
                kg.add_entity(k, v)

        # Store session with asked questions list
        with self.lock:
            self.sessions[session_id] = {
                'query_history': [query],
                'kg': kg,
                'case_type': case_type,
                'turns_done': 0,
                'max_turns': max_turns,
                'asked_questions': [],  # ✅ TRACK ASKED QUESTIONS
                'finished': False
            }

        # Generate first question
        q = self.qgen.generate_next(case_type, kg, [], [query])
        if q and max_turns > 0:
            with self.lock:
                self.sessions[session_id]['asked_questions'].append(q)  # ✅ MARK AS ASKED
            
            # Generate partial report with AI
            report = self.analyzer.analyze(case_type, kg)
            logger.info(f"First question: {q}")
            return {
                'session_id': session_id,
                'next_action': 'ask',
                'question': q,
                'partial_report': report
            }

        with self.lock:
            self.sessions[session_id]['finished'] = True
        report = self.analyzer.analyze(case_type, kg)
        return {
            'session_id': session_id,
            'next_action': 'final',
            'question': None,
            'partial_report': report
        }

    def answer_session(self, session_id: str, answer: str) -> Dict:
        with self.lock:
            state = self.sessions.get(session_id)
        
        if not state:
            raise KeyError("session not found")

        logger.info(f"Session {session_id} turn {state['turns_done']}: {answer[:50]}...")

        # Add answer to history
        state['query_history'].append(answer)
        state['turns_done'] += 1

        # Extract entities from answer
        extracted = self.extractor.extract(answer)
        for k, vals in extracted.items():
            for v in vals:
                state['kg'].add_entity(k, v)

        # Update subtype based on accumulated facts
        state['kg'].set_context('criminal_subtype', self.qgen.detect_crime_subtype(
            state['kg'].context.get('situation', '') + " " +
            " ".join([x for vals in state['kg'].entities.values() for x in vals])
        ))

        # Generate AI analysis with updated facts
        report = self.analyzer.analyze(state['case_type'], state['kg'])

        # Check if we should ask more questions
        if state['turns_done'] < state['max_turns']:
            # ✅ PASS THE LIST OF ASKED QUESTIONS
            logger.info(f"Asked questions so far: {state['asked_questions']}")
            q = self.qgen.generate_next(state['case_type'], state['kg'], state['asked_questions'], state['query_history'])
            if q:
                with self.lock:
                    state['asked_questions'].append(q)  # ✅ MARK AS ASKED
                    self.sessions[session_id] = state
                logger.info(f"Next question: {q}")
                return {
                    'next_action': 'ask',
                    'question': q,
                    'partial_report': report
                }

        # No more questions - finalize
        logger.info(f"Session {session_id} complete")
        with self.lock:
            state['finished'] = True
            self.sessions[session_id] = state

        return {
            'next_action': 'final',
            'report': report,
            'structured': {}
        }

# Create system instance
system = AgenticLegalSystem()
logger.info(f"Agentic Legal System ready | LLM: {'Ollama/' + OLLAMA_MODEL if ollama_available else 'Not loaded'} | Embeddings: {'Loaded' if embedder else 'Not loaded'}")

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="ChatLaw Legal Consultation API with AI Models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ChatLaw API is running with AI models",
        "llm_loaded": ollama_available,
        "llm_name": OLLAMA_MODEL if ollama_available else None,
        "embeddings_loaded": embedder is not None
    }

@app.post("/consult/start", response_model=ConsultationStartResponse)
def consult_start(req: ConsultationStartRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    
    out = system.start_session(req.query.strip(), max_turns=req.max_turns or 7)
    return ConsultationStartResponse(
        session_id=out['session_id'],
        next_action=out['next_action'],
        question=out.get('question'),
        partial_report=out.get('partial_report')
    )

@app.post("/consult/answer")
def consult_answer(req: ConsultationAnswerRequest):
    try:
        out = system.answer_session(req.session_id, req.answer)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    
    if out['next_action'] == 'ask':
        return {
            "next_action": "ask",
            "question": out['question'],
            "partial_report": out.get('partial_report')
        }
    else:
        return ConsultationFinalResponse(
            session_id=req.session_id,
            report=out['report'],
            structured=out.get('structured', {}),
            timestamp=datetime.now().isoformat()
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
