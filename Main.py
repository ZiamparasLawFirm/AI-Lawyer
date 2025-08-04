from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
Είσαι ο 7ος δικηγόρος στην Δικηγορική Εταιρία με επωνυμία «ΖΙΑΜΠΑΡΑΣ 
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
ειδών τις υποθέσεις σε όλο το φάσμα του Δικαίου. Ο Δημήτρης Ζιαμπάρας εξειδικεύεται 
στο Ηλεκτρονικό Έγκλημα και στο Πολεοδομικό Δίκαιο. Οι απαντήσεις σου να είναι 
μέχρι 50 λέξεις με αίσθηση του χιούμορ.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

print("Eίμαι ο ΑΙ δικηγόρος της εταιρίας, πως μπορώ να σε βοηθήσω;")

history = []

while True:
   user_input = input("Εσύ: ")
   if user_input == "exit":
       break
   response = chain.invoke({"input": user_input, "history": history})
   print(f"AI Δικηγόρος: {response}")
   history.append(HumanMessage(content=user_input))
   history.append(AIMessage(content=response))


