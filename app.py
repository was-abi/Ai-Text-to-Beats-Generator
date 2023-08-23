from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small') #4 models available but I used a smaller one because it takes a lot of computationn power
    return model

def generate_music_tensors(description, duration: int):
    print("Describe the type of music you want to create: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]


def save_audio(samples: torch.Tensor):
    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon= "musical_note",
    page_title= "Music Gen"
)

def main():

    st.title("Ai Text to Music-Beats Generator🎵")

    with st.expander("See explanation about the project"):
        st.write("Music Generator app built using Facebook's(Meta's) Audiocraft library. We are using Music Gen Small model. There are four models available but I'll be using the small model because of computing power. If you have a bigger computation power then you might use Music Gen melody as it produces more quality output or accurate data")

    text_area = st.text_area("Enter your description of the music you want to create.......")
    time_slider = st.slider("Select time duration of the audio to be generated(In Seconds)", 0, 30, 5) #arguments are min=0,max=30,default=5

    if text_area and time_slider:
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        print("Musci Tensors: ", music_tensors)
        save_music_file = save_audio(music_tensors)
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    