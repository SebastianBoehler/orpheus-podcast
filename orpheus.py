#!/usr/bin/env python3
"""
Orpheus TTS Module - A modular interface to the Orpheus Text-to-Speech system.
Based on https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
from snac import SNAC

class OrpheusTTS:
    """
    A class to handle text-to-speech generation using the Orpheus model.
    """
    
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-ft", device="cpu"):
        """
        Initialize the Orpheus TTS model.
        
        Args:
            model_name (str): The name of the Orpheus model to use
            device (str): The device to run the model on ("cpu" or "cuda")
        """
        print("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(device)
        
        print("Loading Orpheus model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
    
    def generate_speech(self, prompts, max_new_tokens=1200, temperature=0.6, 
                        top_p=0.95, repetition_penalty=1.1, output_dir=None):
        """
        Generate speech from a list of text prompts.
        
        Args:
            prompts (list): List of text prompts to convert to speech
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Controls randomness in generation (higher = more random)
            top_p (float): Controls diversity via nucleus sampling
            repetition_penalty (float): Penalizes repetition (>= 1.1 required for stable generations)
            output_dir (str, optional): Directory to save audio files. If None, files won't be saved.
            
        Returns:
            list: List of audio samples as numpy arrays
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_samples = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating speech for prompt {i+1}: {prompt}")
            
            # Tokenize input
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            # Add special tokens
            start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
            end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
            
            # Pad input
            padded_tensor = modified_input_ids
            attention_mask = torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)
            
            # Generate output
            print("Generating tokens...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=padded_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
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
            
            # Generate audio
            if code_lists:
                print("Generating audio...")
                samples = self._redistribute_codes(code_lists[0])
                
                # Save audio to file if output_dir is provided
                if output_dir:
                    output_file = os.path.join(output_dir, f"output_{i+1}.wav")
                    sf.write(output_file, samples.detach().squeeze().to("cpu").numpy(), 24000)
                    print(f"Saved audio to {output_file}")
                
                all_samples.append(samples.detach().squeeze().to("cpu").numpy())
            else:
                print(f"Warning: No audio generated for prompt {i+1}")
                all_samples.append(None)
        
        return all_samples
    
    def _redistribute_codes(self, code_list):
        """
        Convert code list to audio using the SNAC model.
        
        Args:
            code_list (list): List of codes to convert to audio
            
        Returns:
            torch.Tensor: Audio samples
        """
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
        audio_hat = self.snac_model.decode(codes)
        return audio_hat


def generate_speech(prompts, output_dir=None, model_name="canopylabs/orpheus-3b-0.1-ft", 
                   device="cpu", max_new_tokens=1200, temperature=0.6, 
                   top_p=0.95, repetition_penalty=1.1):
    """
    Generate speech from a list of text prompts.
    
    Args:
        prompts (list): List of text prompts to convert to speech
        output_dir (str, optional): Directory to save audio files. If None, files won't be saved.
        model_name (str): The name of the Orpheus model to use
        device (str): The device to run the model on ("cpu" or "cuda")
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Controls randomness in generation (higher = more random)
        top_p (float): Controls diversity via nucleus sampling
        repetition_penalty (float): Penalizes repetition (>= 1.1 required for stable generations)
        
    Returns:
        list: List of audio samples as numpy arrays
    """
    tts = OrpheusTTS(model_name=model_name, device=device)
    return tts.generate_speech(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        output_dir=output_dir
    )
