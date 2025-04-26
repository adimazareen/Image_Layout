import cv2
import pandas as pd
import numpy as np
import openai
import os

openai.api_key = "your-api-key"

def generate_caption(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Write a short image caption for: {prompt}"}],
        max_tokens=20
    )
    return response['choices'][0]['message']['content']

def place_image(canvas, img, position, canvas_size=(800, 800)):
    h, w = img.shape[:2]
    if position == "top-left":
        canvas[0:h, 0:w] = img
    elif position == "top-right":
        canvas[0:h, canvas.shape[1]-w:canvas.shape[1]] = img
    elif position == "bottom-left":
        canvas[canvas.shape[0]-h:canvas.shape[0], 0:w] = img
    # Add more positions if needed
    return canvas

def add_caption(canvas, caption, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos_dict = {
        "top-left": (10, 30),
        "top-right": (canvas.shape[1] - 200, 30),
        "bottom-left": (10, canvas.shape[0] - 10),
    }
    cv2.putText(canvas, caption, pos_dict.get(position, (10, 10)), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def create_collage(csv_file):
    df = pd.read_csv(csv_file)
    canvas = np.zeros((800, 800, 3), dtype=np.uint8)  # black canvas

    for index, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        if img is None:
            continue

        img = cv2.resize(img, (200, 200))  # Resize for layout
        caption = row['caption']
        
        # Optional: generate caption using OpenAI
        # caption = generate_caption(row['caption'])

        canvas = place_image(canvas, img, row['position'])
        add_caption(canvas, caption, row['position'])

    cv2.imwrite("output_collage.jpg", canvas)
    print("Collage saved as output_collage.jpg")

# Run
create_collage("sample_images.csv")
