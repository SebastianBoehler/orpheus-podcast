#!/usr/bin/env python3
"""
Gemini Podcast Generator with Orpheus TTS
This script generates a podcast script using Google's Gemini API and then
converts it to speech using the Orpheus TTS system.
"""

import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel

# Quality-ordered voices per language for Orpheus TTS.
# For English, voices are ordered by conversational realism (see Orpheus-TTS README).
# For other languages, see: https://canopylabs.ai/releases/orpheus_can_speak_any_language#info
VOICES = {
    "english": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    "french": ["pierre", "amelie", "marie"],
    "german": ["jana", "thomas", "max"],
    "korean": ["유나", "준서"],
    "hindi": ["ऋतिका"],  # more coming
    "mandarin": ["长乐", "白芷"],
    "spanish": ["javi", "sergio", "maria"],
    "italian": ["pietro", "giulia", "carlo"],
}


class SpeakerTurn(BaseModel):
    name: str
    text: str


def setup_gemini_client():
    """Set up and return a Gemini client using the API key from environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def generate_podcast_script(
    client,
    topic: str = None,
    num_turns: int = 10,
    language: str = "english",
    model_type: str = "finetuned",
):
    """
    Generate a podcast script using Gemini API.

    Args:
        client: Gemini API client
        topic: Optional topic for the podcast
        num_turns: Number of conversation turns to generate
        language: Language for the podcast script (must match a key in VOICES)
        model_type: 'finetuned' or 'pretrained'. For pretrained, always use English voices.

    Returns:
        List of SpeakerTurn objects (name, text)
    """
    model = "gemini-2.5-pro-exp-03-25"
    lang_key = language.lower()
    if model_type == "finetuned":
        voice_list = VOICES.get(lang_key, VOICES["english"])
    else:
        voice_list = VOICES["english"]
    user_prompt = f"""
    Language: {language} \n
    Create a podcast discussing {topic}.
    Generate exactly {num_turns} turns of conversation. Make it engaging, informative, and include some humor.
    Return the result as a list of objects with 'name' and 'text' keys for each turn.
    """
    system_instruction = (
        "You are a professional audio engineer and podcast creator. "
        "Create a natural-sounding conversation. Use hooks and rhetorical questions to keep the audience engaged. "
        f"You can use any of these available voices for {language}: {', '.join(voice_list)} ordered by conversational realism. "
        "For the full list of voices in other languages, see https://canopylabs.ai/releases/orpheus_can_speak_any_language#info. "
        "DO NOT use emotions or tags like <laugh>. "
        "Make use of punctuation and sentence structure. "
        "Use ... to add a pause. "
        "Keep each turn reasonably short (1-3 sentences) due to tts limitations. "
        "To make one person speak longer, chain turns of the same person. "
        "To make it even more natural use some dialect, slang or regional expression now and then. "
        "Feel free to use multiple characters to create an engaging podcast. "
        "Voices go differently well with another, so here are good combinations for English: Zac with Tara, Jess with Leah. "
    )
    contents = [
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=system_instruction)],
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        ),
    ]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_schema": list[SpeakerTurn],
        },
    )
    # Use parsed objects if available, else fallback to raw JSON
    if hasattr(response, "parsed") and response.parsed:
        return response.parsed
    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return []


# All prompt formatting and TTS/audio logic should now be handled in main.py using orpheus.py methods.
