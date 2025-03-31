#!/usr/bin/env python3
"""
Gemini Podcast Generator with Orpheus TTS
This script generates a podcast script using Google's Gemini API and then
converts it to speech using the Orpheus TTS system.
"""

import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import soundfile as sf
from google import genai
from google.genai import types

from orpheus import generate_speech

# Available voices in Orpheus
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
# Available emotions
EMOTIONS = [
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
    "<giggle>",
]


def setup_gemini_client():
    """Set up and return a Gemini client using the API key from environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    return genai.Client(api_key=api_key)


def generate_podcast_script(client, topic: str = None, num_turns: int = 10):
    """
    Generate a podcast script using Gemini API.

    Args:
        client: Gemini API client
        topic: Optional topic for the podcast
        num_turns: Number of conversation turns to generate

    Returns:
        List of dictionaries with speaker name and text
    """
    model = "gemini-2.5-pro-exp-03-25"

    # Create prompt with topic if provided
    user_prompt = f"""
    Create a podcast discussing {topic}.
    Generate exactly {num_turns} turns of conversation. Make it engaging, informative, and include some humor.
    """

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]

    # Create system instruction with available voices
    system_instruction = f"""You are a professional audio engineer and podcast creator. 
    Create a natural-sounding conversation. Use hooks and rhetorical questions to keep the audience engaged.
    You can use any of these available voices: {', '.join(AVAILABLE_VOICES)}

    **DO NOT** use emotions or tags like <laugh>.
    Make use of punctuation and sentence structure.
    Use ... to add a pause.
    Keep each turn reasonably short (1-3 sentences) for better audio flow.
    Feel free to use multiple characters to create an engaging podcast.
    """

    generate_content_config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ],
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["array"],
            properties={
                "array": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["name", "text"],
                        properties={
                            "name": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "text": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=system_instruction),
        ],
    )

    print("Generating podcast script with Gemini...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Parse the JSON response
    try:
        result = json.loads(response.text)
        return result.get("array", [])
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        return []


def prepare_prompts(script: List[Dict[str, str]]) -> List[str]:
    """
    Convert script to Orpheus-compatible prompts.

    Args:
        script: List of dictionaries with speaker name and text

    Returns:
        List of formatted prompts
    """
    prompts = []

    # Map speaker names to available voices
    voice_mapping = {
        "zac": "zac",
        "zack": "zac",
        "zach": "zac",
        "zara": "tara",
        "tara": "tara",
        "leah": "leah",
        "jess": "jess",
        "leo": "leo",
        "dan": "dan",
        "mia": "mia",
        "zoe": "zoe",
    }

    for item in script:
        name = item.get("name", "").lower().strip()
        text = item.get("text", "")

        # Determine the voice to use
        voice = voice_mapping.get(name, "tara")  # Default to tara if name not found

        # Format the prompt
        prompt = f"{voice}: {text}"
        prompts.append(prompt)

    return prompts


def generate_podcast(prompts: List[str], output_dir: str = "generated_audio/podcast"):
    """
    Generate audio for each prompt and save individual files.

    Args:
        prompts: List of formatted prompts
        output_dir: Directory to save audio files

    Returns:
        List of audio samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate speech for each prompt
    print(f"Generating {len(prompts)} audio segments...")
    samples = generate_speech(
        prompts=prompts,
        output_dir=output_dir,
        max_new_tokens=1200,
        temperature=0.6,
        repetition_penalty=1.1,
    )

    print(f"Generated {len(samples)} audio samples")
    return samples


def combine_audio_files(output_dir: str, combined_filename: str = "podcast.wav"):
    """
    Combine all WAV files in the output directory into a single podcast file.

    Args:
        output_dir: Directory containing individual audio files
        combined_filename: Filename for the combined podcast
    """
    # Get all WAV files in the directory
    audio_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".wav")])

    if not audio_files:
        print("No audio files found to combine")
        return

    # Combine audio files
    print(f"Combining {len(audio_files)} audio files...")

    # Read all audio files
    audio_segments = []
    for audio_file in audio_files:
        file_path = os.path.join(output_dir, audio_file)
        data, samplerate = sf.read(file_path)
        audio_segments.append(data)

    # Create a small pause (0.3 seconds of silence)
    pause_duration = int(0.3 * samplerate)
    pause = np.zeros(pause_duration)

    # Combine all segments with pauses in between
    combined = np.array([])
    for segment in audio_segments:
        combined = np.append(combined, segment)
        combined = np.append(combined, pause)

    # Export combined audio
    combined_path = os.path.join(output_dir, combined_filename)
    sf.write(combined_path, combined, samplerate)
    print(f"Combined podcast saved to {combined_path}")


def main():
    """Main function to generate a podcast."""
    parser = argparse.ArgumentParser(
        description="Generate a podcast using Gemini and Orpheus TTS"
    )
    parser.add_argument("--topic", type=str, help="Topic for the podcast", default=None)
    parser.add_argument(
        "--turns", type=int, help="Number of conversation turns", default=10
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory",
        default="generated_audio/podcast",
    )
    args = parser.parse_args()

    # Set up Gemini client
    client = setup_gemini_client()

    # Generate podcast script
    script = generate_podcast_script(client, args.topic, args.turns)

    if not script:
        print("Failed to generate podcast script")
        return

    # Print the generated script
    print("\nGenerated Podcast Script:")
    for i, item in enumerate(script):
        print(f"{i+1}. {item['name']}: {item['text']}")

    # Prepare prompts for Orpheus TTS
    prompts = prepare_prompts(script)

    # Generate audio for each prompt
    generate_podcast(prompts, args.output_dir)

    # Combine audio files into a single podcast
    combine_audio_files(args.output_dir)


if __name__ == "__main__":
    main()
