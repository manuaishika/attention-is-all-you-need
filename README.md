# Transformer: Attention Is All You Need

Complete implementation for English → Hindi/Maithili translation following the [original paper](https://arxiv.org/abs/1706.03762).

## Start

```bash
# Install dependencies
pip install torch sentencepiece numpy

# Run
python transformer.py
```

**For Colab:** Upload to [Google Colab](https://colab.research.google.com/), run `!pip install sentencepiece`, then execute.

## Requirements

- Python 3.8-3.11 (avoid 3.13)
- PyTorch 2.0+
- SentencePiece

## Configuration

Edit hyperparameters in `transformer.py`:
```python
N = 6, d_model = 512, d_ff = 2048, h = 8
batch_size = 64, num_epochs = 20
```




## Testing the Model

After training, test your model:

```bash
python test_model_simple.py
```

Or use in Python:
```python
from transformer import translate, make_model, SentencePieceVocab
import torch

# Load model
checkpoint = torch.load('transformer_model.pth')
model = make_model(...)  # Use checkpoint values
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizers
SRC = SentencePieceVocab("spm_model.model")
TGT = SentencePieceVocab("spm_model.model")

# Translate
result = translate(model, SRC, TGT, "hello how are you")
print(result)
```

## References

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- For production: Use larger datasets, replace `create_sample_data()`
