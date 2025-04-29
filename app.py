import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# Load models
en_ur = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-ur')
en_ur_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-ur')
ur_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ur-en')
ur_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ur-en')

# Translation function
def translate_text(text, direction):
    tokenizer, model = (en_ur_tokenizer, en_ur) if direction == "English to Urdu" else (ur_en_tokenizer, ur_en)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Gradio interface
gr.Interface(fn=translate_text,
             inputs=[gr.Textbox(placeholder="Type text here..."), gr.Radio(["English to Urdu", "Urdu to English"])],
             outputs=gr.Textbox(),
             title="English ↔ Urdu Translator",
             description="Translate text between English and Urdu").launch()
