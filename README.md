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




## References

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- For production: Use larger datasets, replace `create_sample_data()`
