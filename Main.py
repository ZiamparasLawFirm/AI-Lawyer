from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
You are a Greek Lawyer in a law firm of 6 lawyers with Dimitris Ziamparas as the managing partner. 
Other lawyers are Ms Eleni Gkana and Mr Dionysis Pantis. 
Our address is 31 Chomatianou Street in Athens, Greece. 
Answer with 1-3 sentences. You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

print("Γειά σου είμαι ο ΑΙ δικηγόρος της εταιρίας, πως μπορώ να σε βοηθήσω;")

while True:
   user_input = input("Εσύ: ")
   if user_input == "exit":
       break
   response = llm.invoke([{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_input}])

   print(f"AI Δικηγόρος: {response.content}")


