# backend/app/ai_models.py
import openai
from transformers import pipeline
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceModel:
    def __init__(self, task, model_name):
        self.pipeline = pipeline(task, model=model_name)

    def generate_summary(self, text, max_length=150, min_length=40, do_sample=False):
        try:
            summary = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Hugging Face Summarization Error: {e}")
            return "Error generating summary."

    def analyze_sentiment(self, text):
        try:
            sentiment = self.pipeline(text)
            # Assuming binary sentiment for simplicity
            positive = 0
            negative = 0
            for result in sentiment:
                if result['label'] == 'POSITIVE':
                    positive += result['score']
                else:
                    negative += result['score']
            neutral = 1 - (positive + negative)
            return {
                'positive': round(positive, 2),
                'neutral': round(neutral, 2),
                'negative': round(negative, 2)
            }
        except Exception as e:
            logger.error(f"Hugging Face Sentiment Analysis Error: {e}")
            return {'positive': 0, 'neutral': 0, 'negative': 0}

class OpenAIModel:
    def __init__(self, api_key, task):
        self.api_key = api_key
        self.task = task
        openai.api_key = self.api_key

    def paraphrase(self, text):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Paraphrase the following text:\n\n{text}"}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            paraphrased = response['choices'][0]['message']['content'].strip()
            return paraphrased
        except Exception as e:
            logger.error(f"OpenAI Paraphrasing Error: {e}")
            return "Error paraphrasing text."

    def generate_questions(self, text):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Generate a list of insightful questions based on the following text:\n\n{text}"}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            questions_text = response['choices'][0]['message']['content'].strip()
            # Split into individual questions
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            return questions
        except Exception as e:
            logger.error(f"OpenAI Question Generation Error: {e}")
            return ["Error generating questions."]
