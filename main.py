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
from orpheus import generate_speech
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

    # Determine Orpheus model based on language
    LANG_TO_MODEL = {
        "english": "canopylabs/orpheus-3b-0.1-ft",
        "en": "canopylabs/orpheus-3b-0.1-ft",
        "german": "canopylabs/3b-de-ft-research_release",
        "de": "canopylabs/3b-de-ft-research_release",
        "french": "canopylabs/3b-fr-ft-research_release",
        "fr": "canopylabs/3b-fr-ft-research_release",
        "spanish": "canopylabs/3b-es_it-ft-research_release",
        "es": "canopylabs/3b-es_it-ft-research_release",
        "italian": "canopylabs/3b-es_it-ft-research_release",
        "it": "canopylabs/3b-es_it-ft-research_release",
        "korean": "canopylabs/3b-ko-ft-research_release",
        "ko": "canopylabs/3b-ko-ft-research_release",
        "hindi": "canopylabs/3b-hi-ft-research_release",
        "hi": "canopylabs/3b-hi-ft-research_release",
        "chinese": "canopylabs/3b-zh-ft-research_release",
        "zh": "canopylabs/3b-zh-ft-research_release",
    }
    language = args.language.strip().lower()
    model_name = LANG_TO_MODEL.get(language, "canopylabs/orpheus-3b-0.1-ft")

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
