# import streamlit as st
# import os
# import subprocess
# from pathlib import Path
#
# st.set_page_config(page_title="VR180 Video Converter", layout="centered")
#
# st.title("🎥 VR180 Video Converter")
# st.write("Upload a video and convert it into VR180 (SBS) format.")
#
# uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
#
# output_dir = Path("outputs")
# output_dir.mkdir(exist_ok=True)
#
# if uploaded_file is not None:
#     # Save uploaded file
#     input_path = output_dir / uploaded_file.name
#     with open(input_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#
#     st.success(f"✅ Uploaded: {uploaded_file.name}")
#
#     if st.button("Convert to VR180"):
#         st.info("⏳ Processing... this may take some time")
#
#         output_path = output_dir / f"{uploaded_file.name.rsplit('.',1)[0]}_vr180.mp4"
#
#         # Example: Just re-encode + tag metadata for VR180
#         # Replace this with your real processor.py call
#         ffmpeg_cmd = [
#             "ffmpeg", "-i", str(input_path),
#             "-vf", "stereo3d=sbsl:arcd",  # side-by-side to VR180 (example)
#             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
#             str(output_path)
#         ]
#
#         try:
#             subprocess.run(ffmpeg_cmd, check=True)
#             st.success("🎉 Conversion Complete!")
#             st.video(str(output_path))
#             with open(output_path, "rb") as f:
#                 st.download_button("⬇️ Download VR180 Video", f, file_name=output_path.name)
#         except Exception as e:
#             st.error(f"❌ Conversion failed: {e}")

import streamlit as st
from pathlib import Path
import subprocess

# Page config
st.set_page_config(page_title="VR180 Converter", layout="wide")

# Title
st.title("🎥 VR180 3D Video Converter")

st.markdown("""
Upload your video and choose how you want to watch it:  

- **VR180 (SBS):** For VR headsets like Meta Quest, Pico, YouTube VR  
- **Anaglyph (Red-Cyan):** Works with simple red/cyan 3D glasses  
""")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    input_path = Path("input_video.mp4")
    output_sbs = Path("output_sbs.mp4")
    output_anaglyph = Path("output_anaglyph.mp4")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # --- SBS Conversion (VR180) ---
    # if not output_sbs.exists():
    #     st.info("🔄 Converting to Side-by-Side (VR180)...")
    #     subprocess.run([
    #         "ffmpeg", "-i", str(input_path),
    #         "-vf", "scale=iw/2:ih/2,tile=1x2",
    #         "-c:a", "copy", str(output_sbs)
    #     ], check=True)
    if not output_sbs.exists():
        st.info("🔄 Converting to Side-by-Side (VR180)...")
        subprocess.run([
            "ffmpeg", "-i", str(input_path),
            "-filter_complex",
            "[0:v]split=2[left][right]; [left]scale=iw/2:ih[left_scaled]; [right]scale=iw/2:ih[right_scaled]; [left_scaled][right_scaled]hstack=2[v]",
            "-map", "[v]", "-c:a", "copy", str(output_sbs)
        ], check=True)

    # --- Anaglyph Conversion (Red-Cyan) ---
    if not output_anaglyph.exists():
        st.info("🔄 Converting to Anaglyph (Red-Cyan)...")
        subprocess.run([
            "ffmpeg", "-i", str(input_path),
            "-vf", "stereo3d=sbsl:arcd",  # SBS to anaglyph red/cyan
            "-c:a", "copy", str(output_anaglyph)
        ], check=True)

    # Toggle for preview/download
    option = st.radio(
        "Choose Video Output:",
        ["Preview SBS (VR180)", "Preview Anaglyph (Red-Cyan)"]
    )

    if option == "Preview SBS (VR180)":
        st.subheader("🕶️ VR Headset Mode (Side-by-Side)")
        st.video(str(output_sbs))
        st.download_button(
            "⬇️ Download VR180 (SBS)",
            data=open(output_sbs, "rb").read(),
            file_name="vr180_sbs.mp4",
            mime="video/mp4"
        )
        st.caption("SBS is for VR headsets")

    elif option == "Preview Anaglyph (Red-Cyan)":
        st.subheader("👓 Red-Cyan Anaglyph Mode")
        st.video(str(output_anaglyph))
        st.download_button(
            "⬇️ Download Anaglyph (Red-Cyan)",
            data=open(output_anaglyph, "rb").read(),
            file_name="vr180_anaglyph.mp4",
            mime="video/mp4"
        )
        st.caption("Anaglyph works with red-cyan 3D glasses")

