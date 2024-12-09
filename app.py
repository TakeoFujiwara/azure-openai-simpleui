from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from flask_cors import CORS
import os
from openai import AzureOpenAI
import json
import logging

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORSの設定
CORS(app, resources={
    r"/*": {
        "origins": ["http://20.89.65.222", "http://localhost:80"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

@app.after_request
def add_security_headers(response):
    # CSPを一時的に無効化
    response.headers['Content-Security-Policy'] = None
    
    # キャッシュを無効化
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    
    return response

# Azure OpenAI の設定
endpoint = "https://keko-openai-jpe.openai.azure.com/"
deployment = "gpt-4"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")

try:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-02-15-preview"
    )
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        data = request.json
        user_message = data.get('message', '')
        
        if not user_message or not isinstance(user_message, str):
            return jsonify({'error': 'Invalid message format'}), 400

        def generate():
            messages = [
                {
                    "role": "system",
                    "content": "私は親身にアドバイスを提供するAIアシスタントです。利用者の話に耳を傾け、共感的な理解を示しながら、建設的で実践的なサポートを心がけています。"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            try:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7,
                    stream=True
                )

                for chunk in completion:
                    try:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content
                                if content is not None:
                                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                    except AttributeError as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()), 
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Content-Type': 'text/event-stream',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
