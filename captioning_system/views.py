import os
import re
import json
import mimetypes
from django.shortcuts import render
from django.conf import settings
from .models import UploadedImage
from google import genai
from google.genai import types


def get_gemini_client():
    api_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def home(request):
    caption = None
    uploaded_image = None
    similarity = None
    error = None

    if request.method == "POST":
        image_file = request.FILES.get("image")

        if image_file:
            try:
                saved_image = UploadedImage.objects.create(image=image_file)
                uploaded_image = saved_image.image.url

                # Determine MIME type accurately
                mime_type = image_file.content_type
                if not mime_type or mime_type == "application/octet-stream":
                    guessed, _ = mimetypes.guess_type(saved_image.image.path)
                    mime_type = guessed or "image/jpeg"

                with open(saved_image.image.path, "rb") as img_f:
                    image_data = img_f.read()

                client = get_gemini_client()

                prompt = (
                    "Analyze this image carefully. Provide a highly accurate, vivid description "
                    "in one or two sentences. Also compute an accurate match rate confidence score "
                    "as an integer between 0 and 100 based on how comprehensively and precisely "
                    "the description corresponds to the visual elements in the image. "
                    'Return JSON with keys "caption" (string) and "confidence" (integer).'
                )

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )

                raw_text = response.text.strip() if response.text else ""
                # Strip markdown code fences if present
                clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                clean_text = re.sub(r"\s*```$", "", clean_text).strip()

                result = json.loads(clean_text)
                caption = result.get("caption", "No description generated.")
                similarity = result.get("confidence", 95)

            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render(request, "index.html", {
        "caption": caption,
        "uploaded_image": uploaded_image,
        "similarity": similarity,
        "error": error,
    })
