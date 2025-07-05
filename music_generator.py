import streamlit as st
import torch
import torchaudio
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import numpy as np
import time
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading AI model... This may take a few minutes on first run"):
            model_name = "facebook/musicgen-small"
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            st.success(f"✅ Model loaded successfully on {device.upper()}")
            return model, processor, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("💡 Try installing: pip install torch torchaudio transformers")
        st.stop()

def generate_music_local(prompt, duration=10, guidance_scale=3.0):
    try:
        model, processor, device = load_model()
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)
        sample_rate = model.config.audio_encoder.sampling_rate
        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=guidance_scale,
                max_new_tokens=int(duration * 50),
            )
        audio_data = audio_values[0, 0].cpu().numpy()
        audio_data = audio_data / np.max(np.abs(audio_data))
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"❌ Error generating music: {str(e)}")
        return None, None

def audio_to_bytes(audio_data, sample_rate):
    try:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        buffer = io.BytesIO()
        try:
            import scipy.io.wavfile as wavfile
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(buffer, sample_rate, audio_int16)
            buffer.seek(0)
            return buffer.getvalue()
        except ImportError:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            torchaudio.save(tmp_path, audio_tensor, sample_rate)
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            return audio_bytes
    except Exception as e:
        st.error(f"❌ Error converting audio: {str(e)}")
        return None

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #ff4b4b;'>AI Music Generator</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Duration (seconds):", 5, 30, 10, 5)
    with col2:
        guidance_scale = st.slider("Creativity Level:", 1.0, 5.0, 3.0, 0.5)

    st.markdown("### 💬 Need Inspiration?")
    st.markdown("""
    Try prompts like:
    - A happy summer guitar tune
    - Slow romantic piano music
    - Calm ocean ambient vibes
    - High-energy electronic beat
    """)

    st.subheader("🎶 Describe Your Music")
    prompt = st.text_area(
        "Enter your music description:",
        placeholder="e.g., A peaceful piano melody with soft strings, perfect for relaxation...",
        height=100
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎹 Piano Jazz", use_container_width=True):
            st.session_state.example_prompt = "Smooth jazz piano with light drums"
    with col2:
        if st.button("🎸 Acoustic Rock", use_container_width=True):
            st.session_state.example_prompt = "Upbeat acoustic guitar rock song"
    with col3:
        if st.button("🌙 Ambient", use_container_width=True):
            st.session_state.example_prompt = "Peaceful ambient electronic music"

    if hasattr(st.session_state, 'example_prompt'):
        prompt = st.session_state.example_prompt
        st.text_area("Updated prompt:", value=prompt, height=68, disabled=True)

    generate_button = st.button(
        "🎼 Generate Music",
        type="primary",
        use_container_width=True,
        disabled=not prompt.strip()
    )

    if prompt.strip():
        st.info(f"**Prompt:** {prompt}")
        st.info(f"**Duration:** {duration}s | **Creativity:** {guidance_scale}")

    if generate_button and prompt.strip():
        start_time = time.time()
        with st.spinner(f"🎵 Generating {duration}s of music... Please wait"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(20):
                progress_bar.progress((i + 1) * 5)
                if i < 5:
                    status_text.text("🔄 Initializing model...")
                elif i < 15:
                    status_text.text("🎵 Composing music...")
                else:
                    status_text.text("🎧 Finalizing audio...")
                time.sleep(0.1)
            audio_data, sample_rate = generate_music_local(prompt, duration, guidance_scale)
            generation_time = time.time() - start_time
            if audio_data is not None:
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                st.success(f"🎉 Music generated in {generation_time:.1f} seconds!")
                st.subheader("🎧 Your Generated Music")
                audio_bytes = audio_to_bytes(audio_data, sample_rate)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        label="💾 Download Music (WAV)",
                        data=audio_bytes,
                        file_name=f"ai_music_{int(time.time())}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                    with st.expander("📋 Generation Details"):
                        st.write(f"**Prompt:** {prompt}")
                        st.write(f"**Duration:** {duration} seconds")
                        st.write(f"**Sample Rate:** {sample_rate} Hz")
                        st.write(f"**Creativity Level:** {guidance_scale}")
                        st.write(f"**Generation Time:** {generation_time:.1f} seconds")
                        st.write(f"**Model:** facebook/musicgen-small")
                        st.write(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                        st.write(f"**Generated at:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
            progress_bar.empty()
            status_text.empty()

    st.markdown("""
        <div style='text-align: center; padding-top: 30px; font-size: 16px; color: #999;'>
            <p>@Made by Divya</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
