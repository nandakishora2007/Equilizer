import os
import tempfile
from pathlib import Path

import streamlit as st

from agents.coordinator import Coordinator


st.set_page_config(
    page_title="DeepFake Agentic AI",
    page_icon="🔍",
    layout="centered"
)


st.title("🔍 DeepFake Agentic AI")

st.write(
    "Upload a video and let specialized agents independently "
    "analyze visual and audio evidence."
)

st.info(
    "The Face Agent uses the trained EfficientNet deepfake "
    "detector as the primary evidence source."
)


uploaded_file = st.file_uploader(
    "Upload a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True
    ):

        temp_path = None

        try:
            suffix = Path(
                uploaded_file.name
            ).suffix.lower()

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            with st.spinner(
                "Running Face Agent and Audio Agent..."
            ):

                coordinator = Coordinator()

                result = coordinator.analyze(
                    temp_path
                )

            prediction = result["prediction"]
            confidence = float(
                result["confidence"]
            )

            st.divider()

            if prediction == "Fake":

                st.error(
                    f"🚨 DEEPFAKE\n\n"
                    f"Confidence: {confidence:.1f}%"
                )

            elif prediction == "Real":

                st.success(
                    f"✅ REAL\n\n"
                    f"Confidence: {confidence:.1f}%"
                )

            else:

                st.warning(
                    "⚠️ INCONCLUSIVE\n\n"
                    "There was not enough usable evidence."
                )

            st.subheader("Decision Explanation")

            st.write(
                result["explanation"]
            )

            face_result = result["agents"]["Face Agent"]
            audio_result = result["agents"]["Audio Agent"]

            st.subheader("Agent Evidence")

            st.markdown("### 👤 Face Agent")

            st.write(
                f"Prediction: **{face_result['prediction']}**"
            )

            st.write(
                f"Confidence: "
                f"**{face_result['confidence'] * 100:.1f}%**"
            )

            st.write(
                f"Frames analyzed: "
                f"**{face_result['frames_analyzed']}**"
            )

            st.write(
                f"Faces detected: "
                f"**{face_result['faces_detected']}**"
            )

            st.markdown("### 🔊 Audio Agent")

            st.write(
                f"Prediction: **{audio_result['prediction']}**"
            )

            st.write(
                f"Confidence: "
                f"**{audio_result['confidence'] * 100:.1f}%**"
            )

            st.caption(
                audio_result.get("reason", "")
            )

            st.subheader("Agentic Decision Process")

            st.markdown(
                """
                **Face Agent** independently analyzes sampled
                frames using the trained EfficientNet detector.

                **Audio Agent** independently analyzes audio
                characteristics using lightweight features.

                **Fusion Agent** converts both outputs into fake
                probabilities and combines them.

                **Conflict handling** explicitly detects disagreement
                between the agents.

                **Visual evidence receives 70% weight**, while audio
                evidence receives 30%.
                """
            )

        except FileNotFoundError as exc:

            st.error(
                f"Model error:\n\n{exc}"
            )

        except RuntimeError as exc:

            st.error(
                f"Runtime error:\n\n{exc}"
            )

        except ValueError as exc:

            st.error(
                f"Video error:\n\n{exc}"
            )

        except Exception as exc:

            st.error(
                f"Could not analyze the video:\n\n{exc}"
            )

        finally:

            if temp_path is not None:

                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass