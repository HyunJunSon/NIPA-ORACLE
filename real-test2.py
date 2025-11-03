from fastrtc import Stream
import gradio as gr
import numpy as np

def detection(image, slider):
    return np.flip(image, axis=0)

stream = Stream(
    handler=detection,
    modality="video",
    mode="send-receive",
    additional_inputs=[
        gr.Slider(minimum=0, maximum=1, step=0.01, value=0.3)
    ],
)

# HTTPS 설정 (localhost)
stream.ui.launch(
    server_name="localhost",
    server_port=8009,
    ssl_certfile="localhost.pem",
    ssl_keyfile="localhost-key.pem"
)