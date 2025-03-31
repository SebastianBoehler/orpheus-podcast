#!/usr/bin/env python3
"""
Example usage of the Orpheus TTS module
"""

import os
from orpheus import generate_speech

# tara - Best overall voice for general use (default)
# leah, jess, leo, dan, mia, zac, zoe

# emotions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>


def main():
    """
    Demonstrate how to use the Orpheus TTS module
    """
    # Create output directories
    os.makedirs("generated_audio/example1", exist_ok=True)
    os.makedirs("generated_audio/example2", exist_ok=True)

    # Example 1: Using the generate_speech function with default parameters
    print("Example 1: Using default parameters")
    prompts = [
        "tara: Hey there guys. It's, <chuckle> Tara here, and let me introduce you to Zac... who seems to be asleep.",
        "zac: <yawn> Oh, sorry about that. I was up all night coding.",
    ]

    # Generate speech and save to files
    samples = generate_speech(
        prompts=prompts,
        output_dir="generated_audio/example1",
        max_new_tokens=1200,
        temperature=0.6,
        repetition_penalty=1.1,
    )

    print(f"Generated {len(samples)} audio samples")

    # Example 2: Using the same function with different parameters
    print("\nExample 2: Using different parameters")

    # Different prompts with different emotional tags
    emotion_examples = [
        "leah: I just heard the funniest joke ever! <laugh> I can't even repeat it without laughing.",
        "dan: That meeting was so boring. <sigh> I almost fell asleep twice.",
    ]

    # Generate with different parameters
    emotion_samples = generate_speech(
        prompts=emotion_examples,
        output_dir="generated_audio/example2",
        max_new_tokens=1200,
        # Slightly higher temperature for more variation
        temperature=0.7,
        # Higher repetition penalty for faster speech
        repetition_penalty=1.2,
    )

    print(f"Generated {len(emotion_samples)} emotion samples")

    # You can process the audio samples further here
    for i, sample in enumerate(emotion_samples):
        if sample is not None:
            print(f"Sample {i+1} duration: {len(sample)/24000:.2f} seconds")


if __name__ == "__main__":
    main()
