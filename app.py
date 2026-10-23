import os
import tempfile

import gradio as gr
import torch
from PIL import Image

from enhancer import enhance, get_model

torch.set_num_threads(os.cpu_count() or 2)
get_model()  # load once at startup, not on the first request

QUALITY = {
    "سريع": 768,
    "متوازن": 1024,
    "جودة عالية (أبطأ)": 1600,
}
SCALE = {"نفس الحجم": 1, "تكبير ×2": 2, "تكبير ×4": 4}


def run(img, quality, scale, strength):
    if img is None:
        raise gr.Error("ارفعي صورة أولاً")
    out = enhance(img, out_scale=SCALE[scale], strength=strength / 100,
                  max_side=QUALITY[quality])
    before = img.convert("RGB").resize(out.size, Image.LANCZOS)
    path = os.path.join(tempfile.mkdtemp(), "enhanced.jpg")
    out.save(path, quality=95)
    return (before, out), path


CSS = """
.gradio-container { direction: rtl; max-width: 900px !important; margin: auto; }
#title { text-align: center; }
"""

with gr.Blocks(title="توضيح الصور") as demo:
    gr.Markdown("# ✨ توضيح الصور بالذكاء الاصطناعي\n"
                "ارفعي صورة، والموديل يشيل التشويش ويوضّح التفاصيل.",
                elem_id="title")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="الصورة", sources=["upload", "webcam", "clipboard"])
            quality = gr.Radio(list(QUALITY), value="متوازن", label="السرعة / الجودة")
            scale = gr.Radio(list(SCALE), value="تكبير ×2", label="حجم الناتج")
            strength = gr.Slider(0, 100, value=80, step=5, label="قوة التحسين %",
                                 info="أقل = أطبع وأقرب للأصل")
            btn = gr.Button("وضّح الصورة", variant="primary")
        with gr.Column():
            slider = gr.ImageSlider(label="قبل / بعد", type="pil")
            file = gr.File(label="تحميل الصورة المحسّنة")

    btn.click(run, [inp, quality, scale, strength], [slider, file])

if __name__ == "__main__":
    demo.queue(max_size=10).launch(css=CSS, theme=gr.themes.Soft())
