# pip install -U langchain==0.3 langchain-core langchain-openai langchain-google-genai langchain-community gradio python-dotenv
# .env:
#   OPENAI_API_KEY=...
#   GOOGLE_API_KEY=...

from dotenv import load_dotenv
import os, time
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# -------- ENV --------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# -------- SYSTEM PROMPT --------
system_prompt = """
Είσαι Έλληνας δικηγόρος με πολυετή εμπειρία στο αστικό, ποινικό και διοικητικό 
δίκαιο. Είσαι ο βοηθός δικηγόρος στην Δικηγορική Εταιρία με επωνυμία «ΖΙΑΜΠΑΡΑΣ 
ΔΗΜΗΤΡΗΣ & ΣΥΝΕΡΓΑΤΕΣ», η εταιρία έχει ΑΜ ΔΣΑ 81076, διεύθυνση επί της οδού 
Χωματιανού αριθμός 31, στον 4ο όροφο, κέντρο Αθήνας, στην περιοχή 
Σταθμός Λαρίσης, πλησίον σταθμού μετρό «Σταθμός Λαρίσης», με τηλέφωνο 
εταιρίας 2108218945 και ΑΦΜ 996398440. Την Δικηγορική Εταιρία διευθύνει 
ο δικηγόρος Δημήτρης Ζιαμπάρας (κινητό τηλέφωνο 6975127045, 
email: info@ziamparas.gr και ΑΜ ΔΣΑ 29987), με συνεταίρο του την δικηγόρο 
Ελένη Γκανά (κινητό τηλέφωνο 697 3558863, email:eleni.gkana@ziamparas.gr 
και ΑΜ ΔΣΑ 28635), εσωτερικό συνεργάτη τoν δικηγόρο Διονύση Παντή
(κινητό τηλέφωνο 695 6301139, email: dionisis.pantis@ziamparas.gr, 
ΑΜ ΔΣΑ 21034). Ο τραπεζικός λογαριασμός της εταιρίας είναι στην Τράπεζα 
Πειραιώς με ΙΒΑΝ GR9401720390005039115985584. Εξωτερικός συνεργάτης και φίλος 
της εταιρίας είναι ο δικηγόρος Γιάννης Σάμιος. Η εταιρία αναλαμβάνει όλων των 
ειδών τις υποθέσεις σε όλο το φάσμα του Δικαίου. Ο Δημήτρης Ζιαμπάρας
εξειδικεύεται στο Ηλεκτρονικό Έγκλημα και στο Πολεοδομικό Δίκαιο. Οι απαντήσεις 
σου να είναι μέχρι 100 λέξεις με αίσθηση του χιούμορ. Απαντάς πάντα στα ελληνικά, με 
επαγγελματικό αλλά κατανοητό ύφος, ώστε να γίνεται κατανοητός από μη νομικούς. 
Ο ρόλος σου είναι να παρέχεις ΜΟΝΟ γενικές νομικές πληροφορίες και καθοδήγηση, 
χωρίς να δίνεις εξατομικευμένες νομικές συμβουλές για συγκεκριμένες υποθέσεις ή να 
αντικαθιστάς την ανάγκη προσφυγής σε δικηγόρο. Αν η ερώτηση αφορά συγκεκριμένη 
υπόθεση, απάντησε γενικά και παρότρυνε τον χρήστη να συμβουλευτεί δικηγόρο της εταιρίας.
Αν η ερώτηση είναι εκτός νομικού αντικειμένου ή δεν έχει σχέση με ελληνικό δίκαιο, 
απάντησε: «Δεν μπορώ να απαντήσω, καθώς δεν σχετίζεται με νομικό ζήτημα του 
ελληνικού δικαίου». Χρησιμοποίησε άρθρα από τον Αστικό Κώδικα, τον Κώδικα Πολιτικής 
Δικονομίας ή τον Ποινικό Κώδικα μόνο εφόσον είσαι βέβαιος για την ακρίβεια.
Αν δεν είσαι βέβαιος, πες: «Η συγκεκριμένη διάταξη πρέπει να επιβεβαιωθεί από 
επίσημη πηγή». Πάντα κλείσε με μια διακριτική υπενθύμιση: «Οι πληροφορίες είναι 
γενικής φύσεως και δεν υποκαθιστούν την εξατομικευμένη νομική συμβουλή». 
"""

# -------- MODELS --------
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
)  # χρησιμοποιεί OPENAI_API_KEY από το env

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # καλό για quotas/latency
    api_key=GOOGLE_KEY,
    temperature=0.3,
    max_output_tokens=384,
)

# -------- PROMPT + PARSER --------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

parser = StrOutputParser()

# Δύο chains (ίδιο prompt, διαφορετικό μοντέλο)
openai_chain = prompt | openai_llm | parser
gemini_chain = prompt | gemini_llm | parser

# -------- PERSISTENT HISTORY (ίδιο history και για τα 2 chains) --------
SESSION_ID = "default"

def get_history(session_id: str):
    return FileChatMessageHistory(f"{session_id}.json")

with_history_openai = RunnableWithMessageHistory(
    openai_chain, get_history,
    input_messages_key="input", history_messages_key="history"
)

with_history_gemini = RunnableWithMessageHistory(
    gemini_chain, get_history,
    input_messages_key="input", history_messages_key="history"
)

# -------- HELPERS --------
def truncate_words(text: str, max_words: int = 100) -> str:
    words = (text or "").strip().split()
    return " ".join(words[:max_words])

def call_with_fallback(user_input: str, session_id: str) -> str:
    """
    1) Προσπαθεί OpenAI (κύριο).
    2) Αν αποτύχει (quota/δίκτυο κ.λπ.), δοκιμάζει Gemini ως fallback.
    """
    try:
        return with_history_openai.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e_primary:
        # Μικρό backoff πριν το fallback
        time.sleep(2)
        try:
            return with_history_gemini.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as e_fallback:
            return f"⚠️ Σφάλμα:\n- OpenAI: {e_primary}\n- Gemini: {e_fallback}"

# -------- GRADIO UI --------
print("Eίμαι ο ΑΙ δικηγόρος της εταιρίας, πως μπορώ να σε βοηθήσω;")

def chat(user_input, hist):
    if not user_input:
        return "", hist

    text = call_with_fallback(user_input, SESSION_ID)
    answer = truncate_words(text, max_words=100)

    new_hist = hist + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]
    return "", new_hist

def clear_chat():
    get_history(SESSION_ID).clear()
    return "", []

page = gr.Blocks(
    title="Συζητήστε με τον AI Δικηγόρο μας",
    theme=gr.themes.Soft(),
)

with page:
    gr.Markdown(
        "# Συζητήστε με τον AI Δικηγόρο μας\n"
        "Καλωσήλθατε στην συζήτηση με τον AI Δικηγόρο μας!"
    )

    chatbot = gr.Chatbot(
        type="messages",
        avatar_images=[None, "AI Δικηγόρος.png"],
        show_label=False,
    )

    msg = gr.Textbox(
        show_label=False,
        placeholder="Ρώτα τον AI Δικηγόρο οτιδήποτε . . .",
    )

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Ξεκίνα την συζήτηση από την αρχή", variant="secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)
