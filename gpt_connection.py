from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_target_class_from_prompt(prompt, class_name_candidates):
    """
    사용자 입력과 후보 class_name 리스트를 기반으로 GPT에게 강조할 객체 클래스를 물어봅니다.
    """

    system_prompt = (
        "이미지에는 여러 객체가 있으며, 사용자의 요청에 따라 강조할 객체 클래스 하나를 결정합니다.\n"
        f"객체 후보: {', '.join(sorted(set(class_name_candidates)))}\n"
        "반드시 이 후보들 중 하나로만 응답하세요. 일치하는 객체가 없다면 '없음'이라고 응답하세요."
    )
    user_prompt = f"강조할 객체: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()
