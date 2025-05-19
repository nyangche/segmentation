from openai import OpenAI
import openai

openai.api_key = "YOUR_API_KEY"

def get_target_class_from_prompt(prompt):
    system_prompt = "이미지에는 여러 객체가 있으며, 사용자의 요청에 따라 강조할 객체 클래스 하나를 결정합니다."
    user_prompt = f"강조할 객체: {prompt}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message["content"].strip()
