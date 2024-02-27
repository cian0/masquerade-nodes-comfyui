from PIL import Image 
import subprocess 
import os 

p = {
        "text": ("STRING", {"default": "ara ara"}),
        "thickness": ("INT", {"default": 50}),
        "fontSize": ("INT", {"default": 100}),
        "startX": ("INT", {"default": 120}),
        "startY": ("INT", {"default": 120}),
        "fontFace": ("STRING", {"default": "Copyduck.ttf"}),
        "canvasWidth": ("INT", {"default": 512}),
        "canvasHeight": ("INT", {"default": 512}),
        "rotateX": ("FLOAT", {"default": -20.0}),
        "rotateY": ("FLOAT", {"default": 30.0}),
        "rotateZ": ("FLOAT", {"default": -10.0}),
    }


params = [
    p["text"][1]["default"],
    str(p["thickness"][1]["default"]),
    str(p["fontSize"][1]["default"]),
    str(p["startX"][1]["default"]),
    str(p["startY"][1]["default"]),
    p["fontFace"][1]["default"],
    str(p["canvasWidth"][1]["default"]),
    str(p["canvasHeight"][1]["default"]),
    str(p["rotateX"][1]["default"]),
    str(p["rotateY"][1]["default"]),
    str(p["rotateZ"][1]["default"]),
]

script_path = os.path.join(os.getcwd(), "draw3d/drawText.js")

# Execute the JavaScript script with Puppeteer
subprocess.run(["node", script_path] + params, check=True)
