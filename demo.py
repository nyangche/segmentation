import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from main import run_pipeline
from segmentation import draw_focus_overlay_by_class_sam

# GPT API 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GPT로 후보 클래스 목록을 한국어로 번역
def translate_class_list_to_korean(class_name_candidates):
    system_prompt = (
        "다음은 객체 탐지 결과로 나온 클래스 이름입니다. "
        "이 목록을 자연스러운 한국어로 번역해주세요. 단어 사이 띄어쓰기까지 한국어답게 써주세요.\n"
        f"{', '.join(class_name_candidates)}"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "한국어로 번역해주세요"}
        ]
    )

    return response.choices[0].message.content.strip()

# 자연어로 강조할 객체를 입력하면, 원래 class_name을 반환
def get_target_class_from_prompt(prompt, class_name_candidates):
    translated_korean = translate_class_list_to_korean(class_name_candidates)

    system_prompt = (
        "다음은 객체 탐지 결과로 나온 영어 클래스 이름 목록입니다.\n"
        f"객체 후보 목록 (한글): {translated_korean}\n"
    )
    user_prompt = f"강조할 객체: {prompt}"

    # ❌ 사용자 출력 제거
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    result = response.choices[0].message.content.strip().lower()

    for candidate in class_name_candidates:
        if candidate.lower() in result:
            return candidate

    return "none"

# 페이지 설정
st.set_page_config(page_title="Segmentation Demo", layout="centered")
st.title("🤔Perceive Like Humans: Semantic Instance Grouping for Image Segmentation💭")

# 중간 크기의 부제목 + 모델 설명
st.markdown("#### 김민솔 한채헌 최다빈 최혜주")
st.markdown("사용자의 자연어 의도에 따라 중요한 객체를 강조해주는 인간 중심 이미지 세그멘테이션 모델")

# 구분선
st.markdown("---")

# 상태 저장 초기화
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "grouped" not in st.session_state:
    st.session_state.grouped = None
if "instance_mask_array" not in st.session_state:
    st.session_state.instance_mask_array = None
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None
if "class_names" not in st.session_state:
    st.session_state.class_names = None
if "colored_mask" not in st.session_state:
    st.session_state.colored_mask = None
if "color_map" not in st.session_state:
    st.session_state.color_map = None
if "translated_list" not in st.session_state:
    st.session_state.translated_list = None

# 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, f"{timestamp}_{uploaded_file.name}")

    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.image_path = image_path
    st.image(image_path, caption="업로드된 이미지", width=400)

    # 분석 실행 버튼
    if st.button("① 이미지 분석 실행"):
        st.info("이미지를 분석 중입니다...")
        overlay_path, output_dir, grouped, instance_mask_array, colored_mask, color_map = run_pipeline(image_path)

        st.session_state.grouped = grouped
        st.session_state.instance_mask_array = instance_mask_array
        st.session_state.colored_mask = colored_mask
        st.session_state.color_map = color_map
        st.session_state.output_dir = output_dir
        st.session_state.class_names = list(sorted(set(obj["class_name"] for obj in grouped)))
        st.session_state.translated_list = translate_class_list_to_korean(st.session_state.class_names)

        st.success("분석 완료! 🎉")

# 항상 객체 후보 목록 표시
if st.session_state.translated_list:
    st.markdown("**객체 후보 목록:** " + st.session_state.translated_list)

# 강조 요청 입력
if st.session_state.class_names:
    prompt = st.text_input("② 강조할 객체를 입력하세요!")

    if st.button("③ 강조 실행"):
        target_class = get_target_class_from_prompt(prompt, st.session_state.class_names)

        if target_class.lower() not in ["없음", "none", "no"]:
            focus_output_path = os.path.join(st.session_state.output_dir, "focus_overlay.png")
            draw_focus_overlay_by_class_sam(
                st.session_state.image_path,
                st.session_state.instance_mask_array,
                st.session_state.grouped,
                target_class,
                focus_output_path,
            )

            # ✅ 줄바꿈 강조 메시지
            st.success(f"강조 완료! ✅\n\n👉 강조된 객체: `{target_class}`")
            st.image(focus_output_path, caption="강조된 객체 결과", use_container_width=True)
        else:
            st.warning("GPT 판단: 강조할 클래스가 유효하지 않아 강조를 생략합니다.")