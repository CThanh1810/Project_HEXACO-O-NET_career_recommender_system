import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from root .env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

def select_image_via_dialog() -> str:
    """
    Opens a native OS file dialog so the user can easily pick an image with their mouse.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select HEXACO Result Chart Screenshot",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"File dialog error: {e}")
        return ""

def extract_hexaco_from_bytes(img_bytes: bytes, mime_type: str = "image/png") -> list[float]:
    """
    Core function: Uses Gemini Vision to parse 24 HEXACO facet scores from raw image bytes.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found! Please check your .env file.")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=25000))
    img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

    prompt = """You are an expert psychometric OCR assistant.
Extract the 24 HEXACO facet scores from this official test chart image.
The 24 facets must be returned in this EXACT order:
1. Sincerity
2. Fairness
3. Greed Avoidance
4. Modesty
5. Fearfulness
6. Anxiety
7. Dependence
8. Sentimentality
9. Social Self-esteem
10. Social Boldness
11. Sociability
12. Liveliness
13. Forgiveness
14. Gentleness
15. Flexibility
16. Patience
17. Organization
18. Diligence
19. Perfectionism
20. Prudence
21. Aesthetic Appreciation
22. Inquisitiveness
23. Creativity
24. Unconventionality

Note: Ignore the broad domain scores (Honesty-Humility, Emotionality, etc.) and ignore interstitial Altruism. Only extract the 24 specific facet scores.
Return strictly a JSON object:
{"scores": [24 float numbers]}
"""

    models_to_try = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"]
    last_err = None

    for model_name in models_to_try:
        try:
            print(f"   [Vision AI] Analyzing chart via {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[img_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            data = json.loads(response.text)
            scores = data.get("scores", [])
            if len(scores) == 24:
                return [float(s) for s in scores]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to extract scores from image with Gemini Vision. Error: {last_err}")

def extract_hexaco_from_image(image_path: str) -> list[float]:
    """
    Convenience wrapper: Reads image from disk path and calls extract_hexaco_from_bytes.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(path, "rb") as f:
        img_bytes = f.read()

    suffix = path.suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    return extract_hexaco_from_bytes(img_bytes, mime_type=mime_type)
