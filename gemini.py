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

# Only export Gemini logic from this module
VOICES = ["tara", "leah", "zac", "leo", "jess"]


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
    client, topic: str = None, num_turns: int = 10, language: str = "english"
):
    """
    Generate a podcast script using Gemini API.

    Args:
        client: Gemini API client
        topic: Optional topic for the podcast
        num_turns: Number of conversation turns to generate
        language: Language for the podcast script

    Returns:
        List of SpeakerTurn objects (name, text)
    """
    model = "gemini-2.5-pro-exp-03-25"
    user_prompt = f"""
    Language: {language} \n
    Create a podcast discussing {topic}.
    Generate exactly {num_turns} turns of conversation. Make it engaging, informative, and include some humor.
    Return the result as a list of objects with 'name' and 'text' keys for each turn.
    """
    system_instruction = (
        "You are a professional audio engineer and podcast creator. "
        "Create a natural-sounding conversation. Use hooks and rhetorical questions to keep the audience engaged. "
        f"You can use any of these available voices: {', '.join(VOICES)}. "
        "DO NOT use emotions or tags like <laugh>. "
        "Make use of punctuation and sentence structure. "
        "Use ... to add a pause. "
        "Keep each turn reasonably short (1-3 sentences) for better audio flow. ",
        "To make it even more natural use some dialect, slang or regional expression now and then. ",
        "Feel free to use multiple characters to create an engaging podcast."
        "Voices go different well with another, so here are good combinations: ",
        "Zac and tara, jess and lea",
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
        import json

        return json.loads(response.text)
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return []


# All prompt formatting and TTS/audio logic should now be handled in main.py using orpheus.py methods.
