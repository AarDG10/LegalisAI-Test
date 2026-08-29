import streamlit as st
import requests
from googletrans import Translator

translator = Translator()

FASTAPI_URL = "http://127.0.0.1:8000/predict/"

language = st.selectbox("Choose Language", ["English", "Hindi", "Marathi"])


def translate_text(text, dest_language):
    if dest_language == "English":
        return text
    lang_code = {"Hindi": "hi", "Marathi": "mr"}.get(dest_language, "en")
    try:
        return translator.translate(text, dest=lang_code).text
    except Exception:
        return text


st.sidebar.title(translate_text("LegalisAI - Query Interface", language))
model_choice = st.sidebar.selectbox(
    translate_text("Select Model", language), ["legalis", "faq"]
)

st.title(translate_text("LegalisAI - Legal Query Assistant", language))
st.caption(
    translate_text(
        "⚠️ Informational only — not legal advice. Consult a qualified professional for your situation.",
        language,
    )
)

user_input = st.text_area(translate_text("Enter your query:", language), "")


def send_request(text, model_choice):
    payload = {"text": text, "model_choice": model_choice}
    try:
        response = requests.post(FASTAPI_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


if st.button(translate_text("Get Results", language)):
    if not user_input.strip():
        st.warning(translate_text("Please enter a query to proceed.", language))
    else:
        results = send_request(user_input, model_choice)

        if results and results.get("results"):
            st.subheader(
                f"{translate_text('Results from', language)} {results['model']} {translate_text('Model', language)}"
            )
            for idx, result in enumerate(results["results"], 1):
                if model_choice == "legalis":
                    st.markdown(
                        f"### {translate_text('Case', language)} {idx}: {translate_text(result['case_title'], language)}"
                    )
                    st.write(
                        f"**{translate_text('Link', language)}:** [{translate_text('Read More Here...', language)}]({result['case_link']})"
                    )
                    st.write(
                        f"**{translate_text('Similarity Score', language)}:** {round(result['similarity_score'], 2)}"
                    )

                    st.write(
                        f"**{translate_text('Most Similar Sections', language)}:**"
                    )
                    for section in result["sections"]:
                        st.markdown(
                            f"- **{translate_text(section['section_title'], language)}**"
                        )
                        st.write(
                            translate_text(section["section_description"], language)
                        )

                    st.write(f"**{translate_text('Strong Points', language)}:**")
                    for point in result["strong_points"]:
                        st.write(f"- {translate_text(point, language)}")

                    st.write(f"**{translate_text('Weak Points', language)}:**")
                    for point in result["weak_points"]:
                        st.write(f"- {translate_text(point, language)}")

                    st.write("---")

                elif model_choice == "faq":
                    st.write(
                        f"**{translate_text('FAQ', language)} {idx}:** {translate_text(result['faq_prompt'], language)}"
                    )
                    st.write(
                        f"**{translate_text('Answer', language)}:** {translate_text(result['faq_completion'], language)}"
                    )
                    st.write(
                        f"**{translate_text('Similarity Score', language)}:** {round(result['similarity_score'], 2)}"
                    )
                    st.write("---")
        else:
            st.warning(translate_text("No results found.", language))
