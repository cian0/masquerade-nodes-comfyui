import os
import torch
from PIL import Image

from PIL import ImageDraw, ImageFont

import cv2
import numpy as np
import math
from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
import subprocess
import sys

import dlib

DELIMITER = '|'
cached_clipseg_model = None
VERY_BIG_SIZE = 1024 * 1024

package_list = None
def update_package_list():
    import sys
    import subprocess

    global package_list
    package_list = [r.decode().split('==')[0] for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

def ensure_package(package_name, import_path=None):
    global package_list
    if import_path == None:
        import_path = package_name
    if package_list == None:
        update_package_list()

    if package_name not in package_list:
        print("(First Run) Installing missing package %s" % package_name)
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', import_path])
        update_package_list()

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2batch(t: torch.Tensor, bs: torch.Size) -> torch.Tensor:
    if len(t.size()) < len(bs):
        t = t.unsqueeze(3)
    if t.size()[0] < bs[0]:
        t.repeat(bs[0], 1, 1, 1)
    dim = bs[3]
    if dim == 1:
        return tensor2mask(t)
    elif dim == 3:
        return tensor2rgb(t)
    elif dim == 4:
        return tensor2rgba(t)

def tensors2common(t1: torch.Tensor, t2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) < len(t2s):
        t1 = t1.unsqueeze(3)
    elif len(t1s) > len(t2s):
        t2 = t2.unsqueeze(3)

    if len(t1.size()) == 3:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1)
    else:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1, 1)

    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) > 3 and t1s[3] < t2s[3]:
        return tensor2batch(t1, t2s), t2
    elif len(t1s) > 3 and t1s[3] > t2s[3]:
        return t1, tensor2batch(t2, t1s)
    else:
        return t1, t2

class ClipSegNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "precision": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "normalize": (["no", "yes"],),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "get_mask"

    CATEGORY = "Masquerade Nodes"

    def get_mask(self, image, prompt, negative_prompt, precision, normalize):

        model = self.load_model()
        image = tensor2rgb(image)
        B, H, W, _ = image.shape
        # clipseg only works on square images, so we'll just use the larger dimension
        # TODO - Should we pad instead of resize?
        used_dim = max(W, H)

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((used_dim, used_dim), antialias=True) ])
        img = transform(image.permute(0, 3, 1, 2))

        prompts = prompt.split(DELIMITER)
        negative_prompts = negative_prompt.split(DELIMITER) if negative_prompt != '' else []
        with torch.no_grad():
            # Optimize me: Could do positive and negative prompts as part of one batch
            dup_prompts = [item for item in prompts for _ in range(B)]
            preds = model(img.repeat(len(prompts), 1, 1, 1), dup_prompts)[0]
            dup_neg_prompts = [item for item in negative_prompts for _ in range(B)]
            negative_preds = model(img.repeat(len(negative_prompts), 1, 1, 1), dup_neg_prompts)[0] if len(negative_prompts) > 0 else None

        preds = torch.nn.functional.interpolate(preds, size=(H, W), mode='nearest')
        preds = torch.sigmoid(preds)
        preds = preds.reshape(len(prompts), B, H, W)
        mask = torch.max(preds, dim=0).values

        if len(negative_prompts) > 0:
            negative_preds = torch.nn.functional.interpolate(negative_preds, size=(H, W), mode='nearest')
            negative_preds = torch.sigmoid(negative_preds)
            negative_preds = negative_preds.reshape(len(negative_prompts), B, H, W)
            mask_neg = torch.max(negative_preds, dim=0).values
            mask = torch.min(mask, 1. - mask_neg)

        if normalize == "yes":
            mask_min = torch.min(mask)
            mask_max = torch.max(mask)
            mask_range = mask_max - mask_min
            mask = (mask - mask_min) / mask_range
        thresholded = torch.where(mask >= precision, 1., 0.)
        # import code
        # code.interact(local=locals())
        return (thresholded.to(device=image.device), mask.to(device=image.device),)

    def load_model(self):
        global cached_clipseg_model
        if cached_clipseg_model == None:
            ensure_package("clipseg", "clipseg@git+https://github.com/timojl/clipseg.git@bbc86cfbb7e6a47fb6dae47ba01d3e1c2d6158b0")
            from clipseg.clipseg import CLIPDensePredT
            model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            model.eval()

            d64_file = self.download_and_cache('rd64-uni-refined.pth', 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth')
            d16_file = self.download_and_cache('rd16-uni.pth', 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth')
            # Use CUDA if it's available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(d64_file, map_location=device), strict=False)
            model = model.eval().to(device=device)
            cached_clipseg_model = model
        return cached_clipseg_model

    def download_and_cache(self, cache_name, url):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_cache')
        os.makedirs(cache_dir, exist_ok=True)

        file_name = os.path.join(cache_dir, cache_name)
        if not os.path.exists(file_name):
            print(f'Downloading and caching file: {cache_name}')
            with open(file_name, 'wb') as file:
                import requests
                r = requests.get(url, stream=True)
                r.raise_for_status()
                for block in r.iter_content(4096):
                    file.write(block)
            print('Finished downloading.')

        return file_name

# import torch

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


MAX_RESOLUTION=8192

class ImagePadForOutpaintWithPaddingColor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                # Add padding_color as a new parameter
                "padding_color": ("STRING", {"default": "white"})  # Default to white for demonstration
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image"

    def expand_image(self, image, left, top, right, bottom, feathering, padding_color):
        d1, d2, d3, d4 = image.size()

        # Color map for named colors to RGB values
        color_map = {
            'white': (1.0, 1.0, 1.0),
            'black': (0.0, 0.0, 0.0),
            'red': (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue': (0.0, 0.0, 1.0),
            # Add more colors as needed
        }

        # Select the color from the color_map; default to white if not found
        selected_color = color_map.get(padding_color, (1.0, 1.0, 1.0))

        # Create a tensor for the selected color
        color_tensor = torch.tensor(selected_color, dtype=torch.float32).view(1, 1, 1, 3)
        color_tensor = color_tensor.expand(d1, d2 + top + bottom, d3 + left + right, d4)

        # Initialize the new_image tensor with the selected padding color
        new_image = color_tensor.clone()

        # Insert the original image into the new_image tensor
        new_image[:, top:top + d2, left:left + d3, :] = image

        # Mask creation remains unchanged
        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )
        t = torch.zeros((d2, d3), dtype=torch.float32)
        # Feathering logic (unchanged)
        
        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask)

class Generate3DTextImageNode:
    CATEGORY = "Custom"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World"}),
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
                "runningPort": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_3d_text_image"

    def generate_3d_text_image(self, text, thickness, fontSize, startX, startY, fontFace, canvasWidth, canvasHeight, rotateX, rotateY, rotateZ, runningPort):
        # Parameters to pass to the JavaScript script
        params = [
            text,
            str(thickness),
            str(fontSize),
            str(startX),
            str(startY),
            fontFace,
            str(canvasWidth),
            str(canvasHeight),
            str(rotateX),
            str(rotateY),
            str(rotateZ),
            runningPort.strip()
        ]
        
        # Resolve the path to the JavaScript script
        draw_folder = "draw3d"
        draw_file = "drawText.js"
        script_directory = os.path.join(os.path.dirname(__file__), draw_folder)
        script_path = os.path.join(script_directory, draw_file)
        
        # Change the current working directory to the script's directory
        os.chdir(script_directory)
        
        # Execute the JavaScript script with Puppeteer
        subprocess.run(["node", draw_file] + params, check=True)

        # Optionally, change back to the original directory if needed
        # os.chdir(original_directory)
        
        # Resolve the path to the generated image
        image_file = "p5-sketch-dynamic.png"
        image_path = os.path.join(script_directory, image_file)
        
        # Wait for the image file to be available
        # Waiting logic goes here as previously described
        
        # Load the image using Pillow
        image = Image.open(image_path)
        # If you're using pil2tensor, ensure it's defined or imported
        image = pil2tensor(image)
        
        return (image,)

class HelloWorldOverlayText:
    CATEGORY = "Custom"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello World"}),
                "image_width": ("INT", {"default": 800}),
                "image_height": ("INT", {"default": 600}),
                "font_size": ("INT", {"default": 24}),
                "font_color": ("STRING", {"default": "black"}),
                "background_color": ("STRING", {"default": "white"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_overlay_text"

    def draw_overlay_text(self, text, image_width, image_height, font_size, font_color, background_color):
        # Create a new image with the given background color
        image = Image.new("RGB", (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)
        
        # Define the font and calculate text size
        font = ImageFont.truetype("arial.ttf", font_size)
        text_width, text_height = draw.textsize(text, font=font)
        
        # Calculate position and draw the text
        text_x = (image_width - text_width) / 2
        text_y = (image_height - text_height) / 2
        draw.text((text_x, text_y), text, fill=font_color, font=font)
        image = pil2tensor(image)

        return (image,)


class FixHeadRotation:
    def __init__(self):
        # Initialize the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

        print(models_dir)
        self.predictor = dlib.shape_predictor(f'{models_dir}/shape_predictor_68_face_landmarks.dat')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "should_crop": ("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_head_rotation"
    CATEGORY = "Image Analysis"

    def fix_head_rotation(self, image, should_crop):
        # Convert tensor to PIL Image for processing
        image = tensor2pil(image)

        # Convert PIL Image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        height, width = image_cv.shape[:2]
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        angle = 0
        for face in faces:
            landmarks = self.predictor(gray, face)

            # Calculate the angle
            left_eye_point = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye_point = (landmarks.part(45).x, landmarks.part(45).y)
            delta_x = right_eye_point[0] - left_eye_point[0]
            delta_y = right_eye_point[1] - left_eye_point[1]
            angle = np.arctan(delta_y / delta_x) * (180 / np.pi)

        # Rotate and crop the image
        rotated_image_cv = self.rotate_image(image_cv, angle, width, height, should_crop)

        # Convert back to PIL Image
        rotated_image = Image.fromarray(cv2.cvtColor(rotated_image_cv, cv2.COLOR_BGR2RGB))

        # Convert back to tensor
        rotated_image_tensor = pil2tensor(rotated_image)
        return (rotated_image_tensor,)

    def rotate_image(self, image, angle, width, height, should_crop):
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        if should_crop:
            crop_x, crop_y, crop_width, crop_height = self.calculate_crop_dimensions(width, height, angle)
            cropped_image = rotated_image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            return cropped_image
        else: 
            return rotated_image

    @staticmethod
    def calculate_crop_dimensions(width, height, angle):
        angle_rad = math.radians(angle)
        cos_angle = abs(math.cos(angle_rad))
        sin_angle = abs(math.sin(angle_rad))
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        border_horizontal = (new_width - width) / 2
        border_vertical = (new_height - height) / 2

        crop_x = int(border_horizontal)
        crop_y = int(border_vertical)
        crop_width = int(width - (2 * crop_x))
        crop_height = int(height - (2 * crop_y))

        return crop_x, crop_y, crop_width, crop_height


class ResizeToTargetTransparency:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "current_transparency_percentage": ("FLOAT",{"default": 20.0}),
                "target_transparency_percentage": ("FLOAT", {"default": 45.0, "min": 5, "max": 99., "step": 1}),
                "padding_color": (['alpha','white','black','red','green','blue',],),

                # "target_transparency_percentage": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 3., "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE","INT","INT",)
    FUNCTION = "resize_image_to_target_transparency"

    CATEGORY = "Image Analysis"
    
    def resize_image_to_target_transparency(self, image, current_transparency_percentage, target_transparency_percentage, padding_color='alpha'):
        image = tensor2pil(image)  # Assume this converts a tensor to a PIL Image
        
        # Define padding colors
        color_map = {
            'alpha': (0, 0, 0, 0),
            'white': (255, 255, 255, 255),
            'black': (0, 0, 0, 255),
            'red': (255, 0, 0, 255),
            'green': (0, 255, 0, 255),
            'blue': (0, 0, 255, 255)
        }
        
        # Get the RGBA value for the specified padding color
        padding_rgba = color_map.get(padding_color, (0, 0, 0, 0))  # Default to transparent if unknown color
        
        # Load the image
        original_image = image.convert('RGBA')
        original_width, original_height = original_image.size
        
        # Convert to numpy array for efficient processing
        image_data = np.array(original_image)
        alpha_channel = image_data[:, :, 3]  # Extract the alpha channel

        # Count non-transparent pixels efficiently
        non_transparent_pixels = np.count_nonzero(alpha_channel != 0)
        
        # Calculate the desired total area based on the desired transparency percentage
        desired_non_transparent_percentage = 100 - target_transparency_percentage
        desired_total_area = non_transparent_pixels / (desired_non_transparent_percentage / 100)
        total_pixels = original_width * original_height
        scale_factor = desired_total_area / total_pixels
        new_width = int(np.ceil(original_width * np.sqrt(scale_factor)))
        new_height = int(np.ceil(original_height * np.sqrt(scale_factor)))
        
        # Calculate horizontal and vertical expansion
        horizontal_expansion = new_width - original_width
        vertical_expansion = new_height - original_height
        
        # Create a new image with specified padding color
        new_image = Image.new('RGBA', (new_width, new_height), padding_rgba)
        paste_x = horizontal_expansion // 2
        paste_y = vertical_expansion // 2
        new_image.paste(original_image, (paste_x, paste_y))
        
        # Convert the PIL Image back to a tensor
        resized_image = pil2tensor(new_image)

        return (resized_image,horizontal_expansion,vertical_expansion,)
    
    def add_padding_for_target_transparency(self, image, current_transparency_percentage, target_transparency_percentage):
        image = tensor2pil(image)
        # Current dimensions and area
        width, height = image.size
        current_area = width * height

        # Area of transparency
        area_of_transparency = (current_transparency_percentage / 100) * current_area

        # New total area for target transparency
        new_total_area = area_of_transparency / (target_transparency_percentage / 100)

        # Padding area
        padding_area = new_total_area - current_area
        if padding_area < 0:
            raise ValueError("Current transparency is already lower than the target transparency.")

        # Padding dimensions
        padding_width = (new_total_area - current_area) / height
        padding_height = (new_total_area - current_area) / width

        # Padding on each side
        padding_left_right = int(padding_width / 2)
        padding_top_bottom = int(padding_height / 2)

        # Create new image with padding
        new_image = Image.new("RGBA", (width + 2 * padding_left_right, height + 2 * padding_top_bottom), (0, 0, 0, 0))
        new_image.paste(image, (padding_left_right, padding_top_bottom))

        # return new_image
        resized_image = pil2tensor(new_image)

        return (resized_image,padding_width,padding_height,)

class AlphaPercentageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["transparency", "white", "black"],),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calculate_alpha_percentage"

    CATEGORY = "Image Analysis"

    def calculate_alpha_percentage(self, image, mode='transparency'):
        """
        Calculate the percentage based on alpha, white color, or black color in a Pillow image.
        :param image: A Pillow image in RGB or RGBA format.
        :param mode: Mode of calculation ('transparency', 'white', 'black').
        :return: Percentage based on the selected mode.
        """
        image = tensor2pil(image)
        if not isinstance(image, Image.Image):
            raise ValueError("The image must be a Pillow image object.")

        if mode == 'transparency':
            if image.mode != 'RGBA':
                raise ValueError("Image must be in RGBA format for transparency mode.")
            alpha_channel = np.array(image)[:, :, 3]
            percentage = np.mean(alpha_channel / 255) * 100
        elif mode in ['white', 'black']:
            image_rgb = image.convert('RGB')
            data = np.array(image_rgb)
            target_color = [255, 255, 255] if mode == 'white' else [0, 0, 0]
            mask = np.all(data == target_color, axis=-1)
            percentage = np.mean(mask) * 100
        else:
            raise ValueError("Invalid mode. Choose from 'transparency', 'white', 'black'.")

        # return percentage
        return (percentage,)

class MaskMorphologyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "distance": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "op": (["dilate", "erode", "open", "close"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "morph"
    CATEGORY = "Masquerade Nodes"

    def morph(self, image, distance, op):
        # Convert PyTorch tensor to NumPy array (assuming image is a single-channel mask)
        image_np = image.numpy().squeeze()

        # Define the kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (distance * 2 + 1, distance * 2 + 1))

        if op == "dilate":
            result = cv2.dilate(image_np, kernel, iterations=1)
        elif op == "erode":
            result = cv2.erode(image_np, kernel, iterations=1)
        elif op == "open":
            result = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            result = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError("Unsupported operation")

        # Convert the result back to a PyTorch tensor
        result_tensor = torch.from_numpy(result).unsqueeze(0)  # Add batch dimension
        return (result_tensor,)




class MaskCombineOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "op": (["union (max)", "intersection (min)", "difference", "multiply", "multiply_alpha", "add", "greater_or_equal", "greater"],),
                "clamp_result": (["yes", "no"],),
                "round_result": (["no", "yes"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"

    CATEGORY = "Masquerade Nodes"

    def combine(self, image1, image2, op, clamp_result, round_result):
        image1, image2 = tensors2common(image1, image2)

        if op == "union (max)":
            result = torch.max(image1, image2)
        elif op == "intersection (min)":
            result = torch.min(image1, image2)
        elif op == "difference":
            result = image1 - image2
        elif op == "multiply":
            result = image1 * image2
        elif op == "multiply_alpha":
            image1 = tensor2rgba(image1)
            image2 = tensor2mask(image2)
            result = torch.cat((image1[:, :, :, :3], (image1[:, :, :, 3] * image2).unsqueeze(3)), dim=3)
        elif op == "add":
            result = image1 + image2
        elif op == "greater_or_equal":
            result = torch.where(image1 >= image2, 1., 0.)
        elif op == "greater":
            result = torch.where(image1 > image2, 1., 0.)

        if clamp_result == "yes":
            result = torch.min(torch.max(result, torch.tensor(0.)), torch.tensor(1.))
        if round_result == "yes":
            result = torch.round(result)

        return (result,)

class UnaryMaskOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "op": (["invert", "average", "round", "clamp", "abs"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "op_mask"

    CATEGORY = "Masquerade Nodes"

    def op_mask(self, image, op):
        image = tensor2mask(image)
        if op == "invert":
            return (1. - image,)
        elif op == "average":
            mean = torch.mean(torch.mean(image,dim=2),dim=1)
            return (mean.unsqueeze(1).unsqueeze(2).repeat(1, image.shape[1], image.shape[2]),)
        elif op == "round":
            return (torch.round(image),)
        elif op == "clamp":
            return (torch.min(torch.max(image, torch.tensor(0.)), torch.tensor(1.)),)
        elif op == "abs":
            return (torch.abs(image),)

class UnaryImageOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "op": (["invert", "average", "round", "clamp", "abs"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "op_image"

    CATEGORY = "Masquerade Nodes"

    def op_image(self, image, op):
        image = tensor2rgb(image)
        if op == "invert":
            return (1. - image,)
        elif op == "average":
            mean = torch.mean(torch.mean(image,dim=2),dim=1)
            return (mean.unsqueeze(1).unsqueeze(2).repeat(1, image.shape[1], image.shape[2], 1),)
        elif op == "round":
            return (torch.round(image),)
        elif op == "clamp":
            return (torch.min(torch.max(image, torch.tensor(0.)), torch.tensor(1.)),)
        elif op == "abs":
            return (torch.abs(image),)


class BlurNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 10, "min": 0, "max": 48, "step": 1}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 3., "step": 0.01}),
            },
        }

    def gaussian_blur(self, image, kernel_size, sigma):
        # I'll be honest, I'm not sure this calculation is actually correct for a Gaussian blur, but it looks close enough
        kernel = torch.Tensor(kernel_size, kernel_size).to(device=image.device)
        center = kernel_size // 2
        variance = sigma**2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = math.exp(-(x**2 + y**2)/(2*variance))
        kernel /= kernel.sum()

        # Pad the input tensor
        padding = (kernel_size - 1) // 2
        input_pad = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='reflect')

        # Reshape the padded input tensor for batched convolution
        batch_size, num_channels, height, width = image.shape
        input_reshaped = input_pad.reshape(batch_size*num_channels, 1, height+padding*2, width+padding*2)

        # Perform batched convolution with the Gaussian kernel
        output_reshaped = torch.nn.functional.conv2d(input_reshaped, kernel.unsqueeze(0).unsqueeze(0))

        # Reshape the output tensor to its original shape
        output_tensor = output_reshaped.reshape(batch_size, num_channels, height, width)

        return output_tensor

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "Masquerade Nodes"

    def blur(self, image, radius, sigma_factor):
        if len(image.size()) == 3:
            image = image.unsqueeze(3)
        image = image.permute(0, 3, 1, 2)
        kernel_size = radius * 2 + 1
        sigma = sigma_factor * (0.6 * radius - 0.3)
        result = self.gaussian_blur(image, kernel_size, sigma).permute(0, 2, 3, 1)
        if result.size()[3] == 1:
            result = result[:, :, :, 0]
        return (result,)

class ImageToMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["intensity", "alpha"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"

    CATEGORY = "Masquerade Nodes"

    def convert(self, image, method):
        if method == "intensity":
            if len(image.shape) > 3 and image.shape[3] == 4:
                image = tensor2rgb(image)
            return (tensor2mask(image),)
        else:
            return (tensor2rgba(image)[:,:,:,0],)

class MixByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"

    CATEGORY = "Masquerade Nodes"

    def mix(self, image1, image2, mask):
        image1, image2 = tensors2common(image1, image2)
        mask = tensor2batch(mask, image1.size())
        return (image1 * (1. - mask) + image2 * mask,)

class MixColorByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"

    CATEGORY = "Masquerade Nodes"

    def mix(self, image, r, g, b, mask):
        r, g, b = r / 255., g / 255., b / 255.
        image_size = image.size()
        image2 = torch.tensor([r, g, b]).to(device=image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(image_size[0], image_size[1], image_size[2], 1)
        image, image2 = tensors2common(image, image2)
        mask = tensor2batch(tensor2mask(mask), image.size())
        return (image * (1. - mask) + image2 * mask,)

class CreateRectMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["percent", "pixels"],),
                "origin": (["topleft", "bottomleft", "topright", "bottomright"],),
                "x": ("FLOAT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "y": ("FLOAT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "width": ("FLOAT", {"default": 50, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "height": ("FLOAT", {"default": 50, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "image_width": ("INT", {"default": 512, "min": 64, "max": VERY_BIG_SIZE, "step": 64}),
                "image_height": ("INT", {"default": 512, "min": 64, "max": VERY_BIG_SIZE, "step": 64}),
            },
            "optional": {
                "copy_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_mask"

    CATEGORY = "Masquerade Nodes"

    def create_mask(self, mode, origin, x, y, width, height, image_width, image_height, copy_image_size = None):
        min_x = x
        min_y = y

        max_x = min_x + width
        max_y = min_y + height
        if copy_image_size is not None:
            size = copy_image_size.size()
            image_width = size[2]
            image_height = size[1]
        if mode == "percent":
            min_x = min_x / 100.0 * image_width
            max_x = max_x / 100.0 * image_width
            min_y = min_y / 100.0 * image_height
            max_y = max_y / 100.0 * image_height

        if origin == "bottomleft" or origin == "bottomright":
            min_y, max_y = image_height - max_y, image_height - min_y
        if origin == "topright" or origin == "bottomright":
            min_x, max_x = image_width - max_x, image_width - min_x
            
        mask = torch.zeros((image_height, image_width))
        mask[int(min_y):int(max_y)+1, int(min_x):int(max_x)+1] = 1
        return (mask.unsqueeze(0),)

class MaskToRegion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
                "padding": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "constraints": (["keep_ratio", "keep_ratio_divisible", "multiple_of", "ignore"],),
                "constraint_x": ("INT", {"default": 64, "min": 2, "max": VERY_BIG_SIZE, "step": 1}),
                "constraint_y": ("INT", {"default": 64, "min": 2, "max": VERY_BIG_SIZE, "step": 1}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "batch_behavior": (["match_ratio", "match_size"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_region"

    CATEGORY = "Masquerade Nodes"

    def get_region(self, mask, padding, constraints, constraint_x, constraint_y, min_width, min_height, batch_behavior):
        mask = tensor2mask(mask)
        mask_size = mask.size()
        mask_width = int(mask_size[2])
        mask_height = int(mask_size[1])

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[mask_size[0], mask_width * mask_height]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        # Account for padding
        min_x = torch.max(boxes[:,0] - padding, torch.tensor(0.))
        min_y = torch.max(boxes[:,1] - padding, torch.tensor(0.))
        max_x = torch.min(boxes[:,2] + padding, torch.tensor(mask_width))
        max_y = torch.min(boxes[:,3] + padding, torch.tensor(mask_height))

        width = max_x - min_x
        height = max_y - min_y

        # Make sure the width and height are big enough
        target_width = torch.max(width, torch.tensor(min_width))
        target_height = torch.max(height, torch.tensor(min_height))

        if constraints == "keep_ratio":
            target_width = torch.max(target_width, target_height * constraint_x // constraint_y)
            target_height = torch.max(target_height, target_width * constraint_y // constraint_x)
        elif constraints == "keep_ratio_divisible":
            # Probably a more efficient way to do this, but given the bounds it's not too bad
            max_factors = torch.min(constraint_x // target_width, constraint_y // target_height)
            max_factor = int(torch.max(max_factors).item())
            for i in range(1, max_factor+1):
                divisible = constraint_x % i == 0 and constraint_y % i == 0
                if divisible:
                    big_enough = ~torch.lt(target_width, constraint_x // i) * ~torch.lt(target_height, constraint_y // i)
                    target_width[big_enough] = constraint_x // i
                    target_height[big_enough] = constraint_y // i
        elif constraints == "multiple_of":
            target_width[torch.gt(target_width % constraint_x, 0)] = (target_width // constraint_x + 1) * constraint_x
            target_height[torch.gt(target_height % constraint_y, 0)] = (target_height // constraint_y + 1) * constraint_y

        if batch_behavior == "match_size":
            target_width[:] = torch.max(target_width)
            target_height[:] = torch.max(target_height)
        elif batch_behavior == "match_ratio":
            # We'll target the ratio that's closest to 1:1, but don't want to take into account empty masks
            ratios = torch.abs(target_width / target_height - 1)
            ratios[is_empty] = 10000
            match_ratio = torch.min(ratios,dim=0).indices.item()
            target_width = torch.max(target_width, target_height * target_width[match_ratio] // target_height[match_ratio])
            target_height = torch.max(target_height, target_width * target_height[match_ratio] // target_width[match_ratio])

        missing = target_width - width
        min_x = min_x - missing // 2
        max_x = max_x + (missing - missing // 2)

        missing = target_height - height
        min_y = min_y - missing // 2
        max_y = max_y + (missing - missing // 2)

        # Move the region into range if needed
        bad = torch.lt(min_x,0)
        max_x[bad] -= min_x[bad]
        min_x[bad] = 0

        bad = torch.lt(min_y,0)
        max_y[bad] -= min_y[bad]
        min_y[bad] = 0

        bad = torch.gt(max_x, mask_width)
        min_x[bad] -= (max_x[bad] - mask_width)
        max_x[bad] = mask_width

        bad = torch.gt(max_y, mask_height)
        min_y[bad] -= (max_y[bad] - mask_height)
        max_y[bad] = mask_height

        region = torch.zeros((mask_size[0], mask_height, mask_width))
        for i in range(0, mask_size[0]):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                region[i, ymin:ymax+1, xmin:xmax+1] = 1
        return (region,)

class CutByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "force_resize_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "force_resize_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cut"

    CATEGORY = "Masquerade Nodes"

    def cut(self, image, mask, force_resize_width, force_resize_height, mask_mapping_optional = None):
        if len(image.shape) < 4:
            C = 1
        else:
            C = image.shape[3]

        # We operate on RGBA to keep the code clean and then convert back after
        image = tensor2rgba(image)
        mask = tensor2mask(mask)

        if mask_mapping_optional is not None:
            image = image[mask_mapping_optional]

        # Scale the mask to be a matching size if it isn't
        B, H, W, _ = image.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, _, _ = mask.shape

        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, H * W]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        use_width = int(torch.max(width).item())
        use_height = int(torch.max(height).item())

        if force_resize_width > 0:
            use_width = force_resize_width

        if force_resize_height > 0:
            use_height = force_resize_height

        alpha_mask = torch.ones((B, H, W, 4))
        alpha_mask[:,:,:,3] = mask

        image = image * alpha_mask

        result = torch.zeros((B, use_height, use_width, 4))
        for i in range(0, B):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                single = (image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
                resized = torch.nn.functional.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
                result[i] = resized[0]

        # Preserve our type unless we were previously RGB and added non-opaque alpha due to the mask size
        if C == 1:
            return (tensor2mask(result),)
        elif C == 3 and torch.min(result[:,:,:,3]) == 1:
            return (tensor2rgb(result),)
        else:
            return (result,)

class SeparateMaskComponents:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","MASK_MAPPING")
    FUNCTION = "separate"

    CATEGORY = "Masquerade Nodes"

    def separate(self, mask):
        mask = tensor2mask(mask)

        thresholded = torch.gt(mask,0).unsqueeze(1)
        B, H, W = mask.shape
        components = torch.arange(B * H * W, device=mask.device, dtype=mask.dtype).reshape(B, 1, H, W) + 1
        components[~thresholded] = 0

        while True:
            previous_components = components
            components = torch.nn.functional.max_pool2d(components, kernel_size=3, stride=1, padding=1)
            components[~thresholded] = 0
            if torch.equal(previous_components, components):
                break

        components = components.reshape(B, H, W)
        segments = torch.unique(components)
        result = torch.zeros([len(segments) - 1, H, W])
        index = 0
        mapping = torch.zeros([len(segments) - 1], device=mask.device, dtype=torch.int)
        for i in range(len(segments)):
            segment = segments[i].item()
            if segment == 0:
                continue
            image_index = int((segment - 1) // (H * W))
            segment_mask = (components[image_index,:,:] == segment)
            result[index][segment_mask] = mask[image_index][segment_mask]
            mapping[index] = image_index
            index += 1

        return (result,mapping,)


class PasteByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_base": ("IMAGE",),
                "image_to_paste": ("IMAGE",),
                "mask": ("IMAGE",),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],),
                # In the INPUT_TYPES classmethod, add a new optional parameter for the composite technique
                "composite_technique": (["simple", "gaussian", "poisson", "poisson2", "none"],),

            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"

    CATEGORY = "Masquerade Nodes"

    def paste(self, image_base, image_to_paste, mask, resize_behavior, mask_mapping_optional = None, composite_technique = "none"):
        image_base = tensor2rgba(image_base)
        image_to_paste = tensor2rgba(image_to_paste)
        mask = tensor2mask(mask)

        # Scale the mask to be a matching size if it isn't
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]
        if mask_mapping_optional is None:
            if B < PB:
                assert(PB % B == 0)
                image_base = image_base.repeat(PB // B, 1, 1, 1)
            B, H, W, C = image_base.shape
            if MB < B:
                assert(B % MB == 0)
                mask = mask.repeat(B // MB, 1, 1)
            elif B < MB:
                assert(MB % B == 0)
                image_base = image_base.repeat(MB // B, 1, 1, 1)
            if PB < B:
                assert(B % PB == 0)
                image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, MH, MW = mask.shape

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        for i in range(0, MB):
            if is_empty[i]:
                continue
            else:
                image_index = i
                if mask_mapping_optional is not None:
                    image_index = mask_mapping_optional[i].item()
                source_size = image_to_paste.size()
                SB, SH, SW, _ = image_to_paste.shape

                # Figure out the desired size
                width = int(target_width[i].item())
                height = int(target_height[i].item())
                if resize_behavior == "keep_ratio_fill":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        width = int(height * actual_ratio)
                    elif actual_ratio < target_ratio:
                        height = int(width / actual_ratio)
                elif resize_behavior == "keep_ratio_fit":
                    target_ratio = width / height
                    actual_ratio = SW / SH
                    if actual_ratio > target_ratio:
                        height = int(width / actual_ratio)
                    elif actual_ratio < target_ratio:
                        width = int(height * actual_ratio)
                elif resize_behavior == "source_size" or resize_behavior == "source_size_unmasked":
                    width = SW
                    height = SH

                # Resize the image we're pasting if needed
                resized_image = image_to_paste[i].unsqueeze(0)
                if SH != height or SW != width:
                    resized_image = torch.nn.functional.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

                kernel_size = 49
                
                pasting_alpha = mask[i]

                pasting = torch.ones([H, W, C])
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0
                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= (xmax - W)
                    xmax = W
                if ymax > H:
                    source_ymax -= (ymax - H)
                    ymax = H

                # Convert tensor to numpy for Gaussian blur and back
                if composite_technique == 'gaussian':
                    mask_np = mask[i].cpu().numpy()
                    mask_np = cv2.GaussianBlur(255 - mask_np, (kernel_size, kernel_size), kernel_size/3.)
                    mask_np = 255 - mask_np
                    pasting_alpha = torch.from_numpy(mask_np).to(mask.device)

                        # Ensure alpha mask has the same number of channels as the image
                    pasting_alpha = pasting_alpha.unsqueeze(2).repeat(1, 1, C)

                    # Calculate the paste area
                    ymin = max(int(mid_y[i].item() - height / 2), 0)
                    ymax = min(ymin + height, H)
                    xmin = max(int(mid_x[i].item() - width / 2), 0)
                    xmax = min(xmin + width, W)

                    # Ensure resized_image and pasting_alpha are correctly sized
                    resized_image = resized_image[:, :ymax-ymin, :xmax-xmin, :]
                    pasting_alpha = pasting_alpha[ymin:ymax, xmin:xmax, :]

                    # Blend the images
                    result[image_index, ymin:ymax, xmin:xmax, :] = resized_image * pasting_alpha + result[image_index, ymin:ymax, xmin:xmax, :] * (1. - pasting_alpha)
                # ... existing code ...

                elif composite_technique == 'poisson':
                    def calculate_paste_region(center, paste_shape, H, W):
                        paste_height, paste_width, _ = paste_shape
                        center_x, center_y = center

                        # Calculate the top left corner of the paste area
                        xmin = max(center_x - paste_width // 2, 0)
                        ymin = max(center_y - paste_height // 2, 0)

                        # Adjust if the bottom or right edges exceed the base image dimensions
                        xmax = min(xmin + paste_width, W)
                        ymax = min(ymin + paste_height, H)

                        # Adjust xmin and ymin if xmax or ymax adjustments caused the pasted image to shift
                        xmin = max(xmax - paste_width, 0)
                        ymin = max(ymax - paste_height, 0)

                        print(f'Calculated paste region: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}')
                        return ymin, ymax, xmin, xmax
                    
                    # Original mask values are in the range [0, 1]
                    mask_np = mask[i].cpu().numpy()


                    print('Unique mask values before conversion:', np.unique(mask_np))
                    image_base_np = image_base[i].cpu().numpy().astype(np.uint8)
                    image_to_paste_np = image_to_paste[i].cpu().numpy().astype(np.uint8)
                    # mask_np = mask[i].cpu().numpy().astype(np.uint8)
                    # Convert non-zero values to 255 directly without intermediate uint8 conversion
                    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
                    print('Unique mask values after conversion to uint8:', np.unique(mask_np))

                    # Ensure mask is binary for Poisson blending
                    # Convert non-zero mask values to 255 (white) for blending area
                    mask_np[mask_np > 0] = 255
                    print('Mask after conversion to binary for blending:', np.unique(mask_np))

                    # Calculate the center for Poisson blending, ensuring it's within bounds
                    center_x = int(min_x[i].item() + (max_x[i].item() - min_x[i].item()) / 2)
                    center_y = int(min_y[i].item() + (max_y[i].item() - min_y[i].item()) / 2)
                    center = (max(0, min(center_x, W - 1)), max(0, min(center_y, H - 1)))

                    print(f'Calculated center for Poisson blending: center_x={center_x}, center_y={center_y}')

                    # Apply Poisson blending using the corrected mask and center
                    print(f"Base image dimensions: {image_base_np.shape}")
                    print(f"Image to paste dimensions: {image_to_paste_np.shape}")
                    print(f"Mask dimensions (after adding new axis): {mask_np[:, :, np.newaxis].shape}")
                    print(f"Center: {center}")

                    # Convert RGBA to RGB, if necessary
                    image_base_np_rgb = image_base_np[..., :3]  # Drop the alpha channel
                    image_to_paste_np_rgb = image_to_paste_np[..., :3]  # Drop the alpha channel

                    # Use RGB images if conversion is done


                    # Assuming center is (center_x, center_y) and base image dimensions are (H, W)
                    if not (0 <= center_x < W and 0 <= center_y < H):
                        print("Center is out of bounds!")
                    else:
                        print("Center is not out of bounds!")

                    # Calculate the ROI bounds based on the source image dimensions and the center
                    roi_xmin = max(center_x - image_to_paste_np.shape[1] // 2, 0)
                    roi_ymin = max(center_y - image_to_paste_np.shape[0] // 2, 0)
                    roi_xmax = min(roi_xmin + image_to_paste_np.shape[1], W)
                    roi_ymax = min(roi_ymin + image_to_paste_np.shape[0], H)

                    # Check if ROI exceeds base image dimensions
                    if roi_xmax > W or roi_ymax > H:
                        print("ROI exceeds base image dimensions!")
                    else:
                        print("ROI does not exceed base image dimensions!")

                    # Adjust center or source image dimensions if necessary

                    # comp_img_np = cv2.seamlessClone(image_to_paste_np_rgb, image_base_np_rgb, mask_np[:, :, np.newaxis], center, cv2.NORMAL_CLONE)
                        
                        # Create a simple test mask
                    test_mask = np.zeros_like(image_base_np_rgb[:, :, 0])
                    test_center = (100, 100)  # An example simple center
                    cv2.circle(test_mask, test_center, 50, (255), -1)  # Create a circular mask for testing

                    # Try seamlessClone with the simple mask

                    if True:
                        test_comp_img = cv2.seamlessClone(image_to_paste_np_rgb, image_base_np_rgb, test_mask[:, :, np.newaxis], test_center, cv2.NORMAL_CLONE)
                        

                        # comp_img_np = cv2.seamlessClone(image_to_paste_np, image_base_np, mask_np[:, :, np.newaxis], center, cv2.NORMAL_CLONE)

                        # Check if the composite image is not just a black box
                        if np.all(test_comp_img == 0):
                            print('Error: Composite image is entirely black after Poisson blending.')
                        else:
                            print('Poisson blending applied successfully.')

                        # Convert the result back to a tensor
                        comp_img_tensor = torch.from_numpy(test_comp_img).to(image_base.device)

                        # # Update the result image
                        # ymin, ymax, xmin, xmax = calculate_paste_region(center, image_to_paste_np.shape, H, W)  # Implement this function based on your logic
                        # result[i, ymin:ymax, xmin:xmax, :] = comp_img_tensor[ymin:ymax, xmin:xmax, :]

                        # Assuming comp_img_tensor is the RGB result from cv2.seamlessClone
                        # Add an alpha channel indicating full opacity (255)
                        alpha_channel = 255 * np.ones((comp_img_tensor.shape[0], comp_img_tensor.shape[1], 1), dtype=comp_img_tensor.dtype)
                        comp_img_tensor_with_alpha = np.concatenate((comp_img_tensor, alpha_channel), axis=-1)

                        # Now, update the result tensor
                        result[i, ymin:ymax, xmin:xmax, :] = torch.from_numpy(comp_img_tensor_with_alpha).to(image_base.device)


                        print(f'Updated result image for index {i} in the batch.')
                    else:
                        
                        comp_img_np = cv2.seamlessClone(image_to_paste_np, image_base_np, mask_np[:, :, np.newaxis], center, cv2.NORMAL_CLONE)

                        # Check if the composite image is not just a black box
                        if np.all(comp_img_np == 0):
                            print('Error: Composite image is entirely black after Poisson blending.')
                        else:
                            print('Poisson blending applied successfully.')

                        # Convert the result back to a tensor
                        comp_img_tensor = torch.from_numpy(comp_img_np).to(image_base.device)

                        # Update the result image
                        ymin, ymax, xmin, xmax = calculate_paste_region(center, image_to_paste_np.shape, H, W)  # Implement this function based on your logic
                        result[i, ymin:ymax, xmin:xmax, :] = comp_img_tensor[ymin:ymax, xmin:xmax, :]

                        print(f'Updated result image for index {i} in the batch.')


                elif composite_technique == 'poisson2':
                    image_base_np = image_base[i].cpu().numpy().astype(np.uint8)
                    image_to_paste_np = image_to_paste[i].cpu().numpy().astype(np.uint8)
                    mask_np = mask[i].cpu().numpy().astype(np.uint8)

                    center = ((xmin + xmax) // 2, (ymin + ymax) // 2)

                    # Ensure the center is within the image dimensions
                    center = (max(0, min(center[0], image_base_np.shape[1] - 1)), 
                            max(0, min(center[1], image_base_np.shape[0] - 1)))

                    # Apply Poisson blending
                    comp_img_np = cv2.seamlessClone(image_to_paste_np, image_base_np, (mask_np*255)[:,:,np.newaxis], center, cv2.NORMAL_CLONE)

                    # Convert the result back to a tensor
                    comp_img_tensor = torch.from_numpy(comp_img_np).to(image_base.device)

                    # Update the result tensor
                    result[i, ymin:ymax, xmin:xmax, :] = comp_img_tensor[ymin:ymax, xmin:xmax, :]
                
           
                else:
                    pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
                    pasting[:, :, 3] = 1.

                    pasting_alpha = torch.zeros([H, W])
                    pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

                    if resize_behavior == "keep_ratio_fill" or resize_behavior == "source_size_unmasked":
                        # If we explicitly want to fill the area, we are ok with extending outside
                        paste_mask = pasting_alpha.unsqueeze(2).repeat(1, 1, 4)
                    else:
                        paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)

                    result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)

        return (result,)

class GetImageSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT","INT",)
    FUNCTION = "get_size"

    CATEGORY = "Masquerade Nodes"

    def get_size(self, image):
        image_size = image.size()
        image_width = int(image_size[2])
        image_height = int(image_size[1])
        return (image_width, image_height,)

class ChangeChannelCount:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "kind": (["mask", "RGB", "RGBA"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "change_channels"

    CATEGORY = "Masquerade Nodes"

    def change_channels(self, image, kind):
        image_size = image.size()

        if kind == "mask":
            return (tensor2mask(image),)
        elif kind == "RGBA":
            return (tensor2rgba(image),)
        else: # RGB
            return (tensor2rgb(image),)

class ConstantMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.01}),
                "explicit_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "explicit_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
            },
            "optional": {
                "copy_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constant_mask"

    CATEGORY = "Masquerade Nodes"

    def constant_mask(self, value, explicit_height, explicit_width, copy_image_size = None):
        height = explicit_height
        width = explicit_width
        if copy_image_size is not None:
            size = copy_image_size.size()
            height = size[1]
            width = size[2]
        elif explicit_height == 0 or explicit_width == 0:
            # We'll just make a tiny mask and let it get resized by nodes further downstream
            height = 16
            width = 16

        result = torch.zeros([1, height, width])
        result[:,:,:] = value
        return (result,)

class PruneByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prune"

    CATEGORY = "Masquerade Nodes"

    def prune(self, image, mask):
        mask = tensor2mask(mask)

        mean = torch.mean(torch.mean(mask,dim=2),dim=1)
        return (image[mean >= 0.5],)

class MakeImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "append"

    CATEGORY = "Masquerade Nodes"

    def append(self, image1, image2 = None, image3 = None, image4 = None, image5 = None, image6 = None):
        result = image1
        if image2 is not None:
            result = torch.cat((result, image2), 0)
        if image3 is not None:
            result = torch.cat((result, image3), 0)
        if image4 is not None:
            result = torch.cat((result, image4), 0)
        if image5 is not None:
            result = torch.cat((result, image5), 0)
        if image6 is not None:
            result = torch.cat((result, image6), 0)
        return (result,)

class CreateQRCodeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "qr_version": ("INT", {"default": 1, "min": 1, "max": 40, "step": 1}),
                "error_correction": (["L", "M", "Q", "H"], {"default": "H"}),
                "box_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "border": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_qr_code"

    CATEGORY = "Masquerade Nodes"

    def create_qr_code(self, text, size, qr_version, error_correction, box_size, border):
        ensure_package("qrcode")
        import qrcode
        if error_correction =="L":
            error_level = qrcode.constants.ERROR_CORRECT_L
        elif error_correction =="M":
            error_level = qrcode.constants.ERROR_CORRECT_M
        elif error_correction =="Q":
            error_level = qrcode.constants.ERROR_CORRECT_Q
        else:
            error_level = qrcode.constants.ERROR_CORRECT_H

        qr = qrcode.QRCode(
                version=qr_version,
                error_correction=error_level,
                box_size=box_size,
                border=border)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size,size))
        # Convert img (a PIL Image) into a torch tensor
        tensor = torch.from_numpy(np.array(img))
        return (tensor2rgb(tensor.unsqueeze(0)),)

def rgb2hsv(rgb):
    # rgb is a tensor in the form [B, H, W, C]
    r = rgb[...,0]
    g = rgb[...,1]
    b = rgb[...,2]

    hsv = torch.zeros_like(rgb)
    hsv_h = hsv[...,0]
    hsv_s = hsv[...,1]
    hsv_v = hsv[...,2]

    # Value
    hsv_v[:], max_idx = torch.max(rgb, dim=3)

    chroma = hsv_v - torch.min(rgb, dim=3).values

    # Hue
    sixth = 1.0 / 6.0
    hsv_h[max_idx == 0] = (sixth * ((g - b) / chroma % 6))[max_idx == 0]
    hsv_h[max_idx == 1] = (sixth * ((b - r) / chroma + 2))[max_idx == 1]
    hsv_h[max_idx == 2] = (sixth * ((r - g) / chroma + 4))[max_idx == 2]
    hsv_h[chroma == 0] = 0

    # Saturation
    hsv_s[chroma != 0] = chroma[chroma != 0] / hsv_v[chroma != 0]

    return hsv

def hsv2rgb(hsv):
    # hsv is a tensor in the form [B, H, W, C] where C is (h,s,v)
    h = hsv[...,0]
    h = h % 1.0
    s = hsv[...,1]
    v = hsv[...,2]

    rgb = torch.zeros_like(hsv)

    chroma = v * s
    hp = (h * 6.0).type(torch.uint8)
    x = chroma * (1 - torch.abs((h * 6.0) % 2 - 1))

    zeros = torch.zeros_like(x)
    rgb[hp == 0] = torch.stack([chroma, x, zeros], dim=3)[hp == 0]
    rgb[hp == 1] = torch.stack([x, chroma, zeros], dim=3)[hp == 1]
    rgb[hp == 2] = torch.stack([zeros, chroma, x], dim=3)[hp == 2]
    rgb[hp == 3] = torch.stack([zeros, x, chroma], dim=3)[hp == 3]
    rgb[hp == 4] = torch.stack([x, zeros, chroma], dim=3)[hp == 4]
    rgb[hp == 5] = torch.stack([chroma, zeros, x], dim=3)[hp == 5]

    rgb += (v - chroma).unsqueeze(3).repeat(1,1,1,3)
    return rgb

def hsv2hsl(hsv):
    hsl = torch.zeros_like(hsv)
    h = hsv[...,0]
    s = hsv[...,1]
    v = hsv[...,2]

    hsl[...,0] = h
    hsl[...,2] = v * (1. - s / 2.)
    l = hsl[...,2]
    defined = (l != 0) & (l != 1)
    hsl[...,1][defined] = ((v - l) / torch.min(l, 1. - l))[defined]
    return hsl

def hsl2hsv(hsl):
    hsv = torch.zeros_like(hsl)
    h = hsl[...,0]
    s = hsl[...,1]
    l = hsl[...,2]

    hsv[...,0] = h
    hsv[...,2] = l + s * torch.min(l, 1. - l)
    v = hsv[...,2]
    defined = (v != 0)
    hsv[...,1][defined] = (2. * (1. - l / v))[defined]
    return hsv

class ConvertColorSpace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "in_space": (["RGB", "HSV", "HSL"],),
                "out_space": (["RGB", "HSV", "HSL"],),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_color_space"

    CATEGORY = "Masquerade Nodes"

    def convert_color_space(self, in_space, out_space, image):
        if in_space == out_space:
            return (image,)

        image = tensor2rgb(image)

        if in_space == "HSV":
            hsv = image
        if in_space == "RGB":
            hsv = rgb2hsv(image)
        elif in_space == "HSL":
            hsv = hsl2hsv(image)

        # We are now in RGB or HSV
        if out_space == "HSV":
            return (hsv,)
        elif out_space == "RGB":
            return (hsv2rgb(hsv),)
        else:
            assert out_space == "HSL"
            return (hsv2hsl(hsv),)

class MaqueradeIncrementerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "max_value": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "increment"

    CATEGORY = "Masquerade Nodes"

    def increment(self, seed, max_value):
        return (seed % max_value,)


NODE_CLASS_MAPPINGS = {
    "Mask By Text": ClipSegNode,
    "Mask Morphology": MaskMorphologyNode,
    "Alpha Percentage": AlphaPercentageNode,    
    "ResizeToTargetTransparency": ResizeToTargetTransparency,
    "ImagePadForOutpaintWithPaddingColor": ImagePadForOutpaintWithPaddingColor,
    "Combine Masks": MaskCombineOp,
    "Unary Mask Op": UnaryMaskOp,
    "Unary Image Op": UnaryImageOp,
    "Blur": BlurNode,
    "Image To Mask": ImageToMask,
    "Mix Images By Mask": MixByMask,
    "Mix Color By Mask": MixColorByMask,
    "Mask To Region": MaskToRegion,
    "Cut By Mask": CutByMask,
    "Paste By Mask": PasteByMask,
    "Get Image Size": GetImageSize,
    "Change Channel Count": ChangeChannelCount,
    "Constant Mask": ConstantMask,
    "Prune By Mask": PruneByMask,
    "Separate Mask Components": SeparateMaskComponents,
    "Create Rect Mask": CreateRectMask,
    "Make Image Batch": MakeImageBatch,
    "Create QR Code": CreateQRCodeNode,
    "Convert Color Space": ConvertColorSpace,
    "MasqueradeIncrementer": MaqueradeIncrementerNode,
    "FixHeadRotation": FixHeadRotation,
    "HelloWorldOverlayText": HelloWorldOverlayText,
    "Generate3DTextImageNode": Generate3DTextImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask By Text": "Mask By Text",
    "Mask Morphology": "Mask Morphology",
    "Combine Masks": "Combine Masks",
    "Unary Mask Op": "Unary Mask Op",
    "Unary Image Op": "Unary Image Op",
    "Blur": "Blur",
    "Image To Mask": "Image To Mask",
    "Mix Images By Mask": "Mix Images By Mask",
    "Mix Color By Mask": "Mix Color By Mask",
    "Mask To Region": "Mask To Region",
    "Cut By Mask": "Cut By Mask",
    "Paste By Mask": "Paste By Mask",
    "Get Image Size": "Get Image Size",
    "Change Channel Count": "Change Channel Count",
    "Constant Mask": "Constant Mask",
    "Prune By Mask": "Prune By Mask",
    "Separate Mask Components": "Separate Mask Components",
    "Create Rect Mask": "Create Rect Mask",
    "Make Image Batch": "Make Image Batch",
    "Create QR Code": "Create QR Code",
    "Convert Color Space": "Convert Color Space",
    "MasqueradeIncrementer": "Incrementer",
    "ImagePadForOutpaintWithPaddingColor": "Image Pad For Outpaint With Padding Color",
    "FixHeadRotation": "Fix Head Rotation",
    "HelloWorldOverlayText": "Hello World Overlay Text",
    "Generate3DTextImageNode": "Generate 3D Text Image Node",
}
