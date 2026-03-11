# Ground Rules for Teammates

Welcome to the project! As you contribute, please strictly adhere to the following ground rules established for the team:

## 1. The `.gitignore` Rule
**Never push massive data or model files to GitHub.** 
You must keep your large files strictly on your local machines. This includes, but is not limited to:
- Massive CSV files (`*.csv`)
- PyTorch tensor files (`*.pt`, `*.pth`)
- Standard model checkpoints (`*.h5`, `*.ckpt`, `*.safetensors`, `*.bin`, `*.weights.h5`)

Our root `.gitignore` is already configured to ignore these file extensions. Please do not bypass these rules.

## 2. The `requirements.txt` Rule
**Use the master `requirements.txt` file.**
If you need a new Python library for your module (e.g., `nltk` or `spacy`), you should add it to the master `requirements.txt` file in the root directory. 
- Do **not** create your own requirement files nested within your module's directory. 
- Keeping all dependencies centralized ensures everyone uses a consistent environment.
