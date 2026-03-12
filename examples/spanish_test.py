import argparse

import torch
import torchaudio

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TADA with a Spanish prompt audio clip.")
    parser.add_argument("--audio", required=True, help="Path to the Spanish prompt wav file.")
    parser.add_argument(
        "--prompt-text",
        default="Hoy fui al mercado y encontre frutas frescas a buen precio.",
        help="Transcript for the Spanish prompt audio used for forced alignment.",
    )
    parser.add_argument(
        "--target-text",
        default="Hoy es un buen dia para salir a caminar por el parque.",
        help="Text to generate from the model.",
    )
    parser.add_argument("--encoder-model", default="HumeAI/tada-codec")
    parser.add_argument("--lm-model", default="HumeAI/tada-3b-ml")
    parser.add_argument("--language", default="es")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder.from_pretrained(
        args.encoder_model,
        subfolder="encoder",
        language=args.language,
    ).to(device)

    model = TadaForCausalLM.from_pretrained(args.lm_model).to(device)

    audio, sample_rate = torchaudio.load(args.audio)
    audio = audio.to(device)

    # Provide transcript for non-English prompts so encoder uses forced alignment.
    prompt = encoder(audio, text=[args.prompt_text], sample_rate=sample_rate)

    output = model.generate(
        prompt=prompt,
        text=args.target_text,
    )

    print("Device:", device)
    print("Model output:")
    print(output)


if __name__ == "__main__":
    main()