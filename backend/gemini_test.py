from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

response = client.models.generate_content_stream(
    model='gemini-2.0-flash',
    contents='write a short poem'
)
for chunk in response:
    print(chunk.text, end='', flush=True)
print()  # newline at end