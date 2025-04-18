# Orpheus TTS Podcast Generator

This project leverages [Canopy AI's Orpheus TTS](https://canopylabs.ai/model-releases) model to generate high-quality, multi-speaker podcasts. It integrates with Gemini to create engaging and dynamic conversational audio.

## Features

- **Multi-Speaker Support:** Generate conversations between multiple characters with distinct voices.
- **Gemini Integration:** Utilizes Gemini to create podcast scripts and conversational content.
- **Emotive Speech:** Orpheus TTS supports emotive tags for realistic and engaging delivery.
- **High-Quality Audio:** Produces clear and natural-sounding audio at 24kHz.
- **Multi-Language Support:** Supports multiple languages (English, German, French, Spanish, Italian, Korean, Hindi, Chinese).

---

**Demo:** [sebastianboehler.github.io/orpheus-podcast](https://sebastianboehler.github.io/orpheus-podcast/)

## License & Model Terms

> **Note:** This project uses the Orpheus TTS models from Canopy AI. You must respect and comply with the license terms and usage restrictions provided by Canopy Labs for the Orpheus models. See the [official release page](https://canopylabs.ai/model-releases) for details.

## Model Selection: Finetuned vs Pretrained

You can choose between **finetuned** and **pretrained** Orpheus TTS models for each language.

- **Pretrained models are recommended for best quality** and are the default.
- Finetuned models support language-specific voices.
- Pretrained models use English voices for all languages, but produce more natural and high-quality speech.

To select the model type, use the `--model-type` argument or set the `PODCAST_MODEL_TYPE` environment variable.

Example:

```sh
python main.py --language german --model-type pretrained
```

or

```sh
export PODCAST_MODEL_TYPE=finetuned
python main.py --language french
```

## Usage

1.  **Setup:** Ensure you have the required dependencies installed (see `requirements.txt`).
2.  **Configuration:** Configure the script with your desired speaker voices and Gemini prompts.
3.  **Generation:** Run the script to generate your podcast audio.
