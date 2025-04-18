#!/usr/bin/env python3
"""
Main Podcast Generator Script
Orchestrates podcast generation using Gemini (for script) and Orpheus TTS (for audio).
"""

from dotenv import load_dotenv
import os
load_dotenv()

import argparse
from gemini import setup_gemini_client, generate_podcast_script
from orpheus import generate_speech, LANG_TO_MODEL
from utils import combine_audio_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate a podcast using Gemini and Orpheus TTS"
    )
    parser.add_argument("--topic", type=str, default=os.environ.get("PODCAST_TOPIC", None), help="Podcast topic")
    parser.add_argument("--turns", type=int, default=int(os.environ.get("PODCAST_TURNS", 8)), help="Number of conversation turns")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory",
        default="generated_audio/podcast",
    )
    parser.add_argument("--language", type=str, default=os.environ.get("PODCAST_LANGUAGE", "english"), help="Language for the podcast script")
    parser.add_argument("--model-type", type=str, default=os.environ.get("PODCAST_MODEL_TYPE", "pretrained"), help="Model type for Orpheus TTS")
    args = parser.parse_args()

    # Set up Gemini client
    client = setup_gemini_client()

    # Generate podcast script
    script = generate_podcast_script(client, args.topic, args.turns, args.language)

    if not script:
        print("Failed to generate podcast script")
        return

    # Print the generated script
    print("\nGenerated Podcast Script:")
    for i, item in enumerate(script):
        print(f"{i+1}. {item.name}: {item.text}")

    # Prepare prompts for Orpheus TTS
    prompts = [f"{item.name}: {item.text}" for item in script]

    # Determine Orpheus model based on language and user preference (default: pretrained)
    language = args.language.strip().lower()
    model_type = getattr(args, "model_type", "pretrained")  # expects 'pretrained' or 'finetuned'
    language_models = LANG_TO_MODEL.get(language, LANG_TO_MODEL["english"])
    model_name = language_models.get(model_type, language_models["pretrained"])

    # Generate audio for each prompt using orpheus
    generate_speech(
        prompts=prompts,
        output_dir=args.output_dir,
        model_name=model_name,
        max_new_tokens=1200,
        temperature=0.6,
        repetition_penalty=1.1,
    )

    # Combine audio files into a single podcast
    combine_audio_files(args.output_dir)


if __name__ == "__main__":
    main()
