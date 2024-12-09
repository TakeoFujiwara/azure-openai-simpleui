# app.py
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import os
from openai import AzureOpenAI
import json
import logging

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Azure OpenAI の設定
endpoint = os.getenv("ENDPOINT_URL", "https://keko-openai-jpe.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")

# 環境変数のログ出力（本番環境では注意）
logger.debug(f"Endpoint: {endpoint}")
logger.debug(f"Deployment: {deployment}")
logger.debug(f"API Key set: {'Yes' if subscription_key != 'REPLACE_WITH_YOUR_KEY_VALUE_HERE' else 'No'}")

try:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

# app.py の chat 関数部分を更新
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        logger.debug(f"Received message: {user_message}")

        def generate():
            messages = [
                {
                    "role": "system",
                    "content": "人生相談の専門家として相談相手の相談に役立つ AI アシスタントです。"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            try:
                logger.debug("Sending request to Azure OpenAI")
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=True
                )

                for chunk in completion:
                    try:
                        # チャンクの内容をログに出力
                        logger.debug(f"Raw chunk: {chunk}")
                        
                        # choices が存在し、かつ空でないことを確認
                        if hasattr(chunk, 'choices') and chunk.choices:
                            # delta が存在し、content が None でないことを確認
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content
                                if content is not None:
                                    logger.debug(f"Processed content: {content}")
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                    except AttributeError as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)