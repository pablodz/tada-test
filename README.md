# tada-test

Devcontainer-ready workspace for running HumeAI TADA inference with PyTorch and Torchaudio.

## Included

- CUDA-capable devcontainer configuration
- Python dependencies for TADA + audio loading
- Spanish test script based on your sample flow

## Open In Devcontainer

1. Open this folder in VS Code.
2. Run "Dev Containers: Reopen in Container".
3. Wait for dependencies to install from `requirements.txt`.

## Run The Spanish Test

Place a Spanish prompt clip at `samples/es_prompt.wav`, then run:

```bash
python examples/spanish_test.py --audio samples/es_prompt.wav
```

Optional flags:

```bash
python examples/spanish_test.py \
	--audio samples/es_prompt.wav \
	--prompt-text "Hoy fuimos al centro y habia mucha gente en la plaza" \
	--target-text "Hoy es un gran dia para aprender algo nuevo."
```
