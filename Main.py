# pip install -U langchain==0.3 langchain-core langchain-google-genai gradio python-dotenv
# .env -> GOOGLE_API_KEY=...

from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
gemini_key = os.getenv("GOOGLE_API_KEY")

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

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_key,
    temperature=0.5,
)

print("Eίμαι ο ΑΙ δικηγόρος της εταιρίας, πως μπορώ να σε βοηθήσω;")

def chat(user_input, hist):
    if not user_input:
        return "", hist

    # Χτίζουμε την πλήρη λίστα μηνυμάτων: System → (history) → Human
    messages = [SystemMessage(content=system_prompt)]

    for item in hist or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_input))

    # Κλήση μοντέλου: επιστρέφει AIMessage
    ai_msg = llm.invoke(messages)
    answer = (ai_msg.content or "").strip()

    new_hist = hist + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]
    return "", new_hist

def clear_chat():
    return "", []

page = gr.Blocks(
    title="Συζητήστε με τον AI Δικηγόρο μας",
    theme=gr.themes.Soft(),
)

with page:
    gr.Markdown(
        "# Συζητήστε με τον AI Δικηγόρο μας\nΚαλωσήλθατε στην συζήτηση με τον AI Δικηγόρο μας!"
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
