#!/usr/bin/env python3
# based on https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
from snac import SNAC


def main():
    # Create output directory
    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SNAC model...")
    # Load SNAC model for audio decoding
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cpu")  # Use CPU for compatibility

    print("Loading Orpheus model...")
    # Model name
    model_name = "canopylabs/orpheus-3b-0.1-ft"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to("cpu")  # Use CPU for compatibility
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define prompts
    # <giggle> seems to produce a sigh
    # can: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
    prompts = [
        "tara: Hey there guys. It's, <chuckle> Tara here, and let me introduce you to Zac... who seems to be asleep."
    ]

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating speech for prompt {i+1}: {prompt}")

        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Add special tokens
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor(
            [[128009, 128260]], dtype=torch.int64
        )  # End of text, End of human
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

        # Pad input
        padding = 0  # No padding needed for a single input
        padded_tensor = modified_input_ids
        attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)

        # Generate output
        print("Generating tokens...")
        # repetition_penalty>=1.1 is required for stable generations.
        # Increasing repetition_penalty and temperature makes the model speak faster
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=padded_tensor,
                attention_mask=attention_mask,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=128258,
            )

        # Parse output as speech
        print("Processing generated tokens...")
        token_to_find = 128257
        token_to_remove = 128258

        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
        else:
            cropped_tensor = generated_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        # Convert codes to audio
        def redistribute_codes(code_list):
            layer_1 = []
            layer_2 = []
            layer_3 = []
            for i in range((len(code_list) + 1) // 7):
                layer_1.append(code_list[7 * i])
                layer_2.append(code_list[7 * i + 1] - 4096)
                layer_3.append(code_list[7 * i + 2] - (2 * 4096))
                layer_3.append(code_list[7 * i + 3] - (3 * 4096))
                layer_2.append(code_list[7 * i + 4] - (4 * 4096))
                layer_3.append(code_list[7 * i + 5] - (5 * 4096))
                layer_3.append(code_list[7 * i + 6] - (6 * 4096))
            codes = [
                torch.tensor(layer_1).unsqueeze(0),
                torch.tensor(layer_2).unsqueeze(0),
                torch.tensor(layer_3).unsqueeze(0),
            ]
            audio_hat = snac_model.decode(codes)
            return audio_hat

        # Generate and save audio
        if code_lists:
            print("Generating audio...")
            samples = redistribute_codes(code_lists[0])

            # Save audio to file
            output_file = os.path.join(output_dir, f"output_{i+1}.wav")
            sf.write(output_file, samples.detach().squeeze().to("cpu").numpy(), 24000)
            print(f"Saved audio to {output_file}")
        else:
            print(f"Warning: No audio generated for prompt {i+1}")

    print("\nAll audio files generated successfully!")
    print(f"Audio files are saved in the '{output_dir}' directory")


if __name__ == "__main__":
    main()
