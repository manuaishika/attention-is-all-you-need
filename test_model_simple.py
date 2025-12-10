"""
Simple test script for the trained Transformer model
"""

import torch
from transformer import make_model, translate, SentencePieceVocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model and tokenizers...")

# Load model
checkpoint = torch.load('transformer_model.pth', map_location=device)
model = make_model(
    checkpoint['src_vocab_size'],
    checkpoint['tgt_vocab_size'],
    N=checkpoint['N'],
    d_model=checkpoint['d_model'],
    d_ff=checkpoint['d_ff'],
    h=checkpoint['h'],
    dropout=checkpoint['dropout']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizers
SRC = SentencePieceVocab("spm_model.model")
TGT = SentencePieceVocab("spm_model.model")

print("Model loaded! Testing translations...\n")
print("=" * 60)

test_sentences = [
    "hello how are you",
    "what is your name",
    "i am a student",
    "thank you very much",
    "good morning"
]

for sent in test_sentences:
    translation = translate(model, SRC, TGT, sent)
    print(f"English: {sent}")
    try:
        print(f"Translation: {translation}")
    except UnicodeEncodeError:
        # Save to file if console can't display
        with open("translations.txt", "a", encoding="utf-8") as f:
            f.write(f"{sent} -> {translation}\n")
        print(f"Translation: [Saved to translations.txt]")
    print("-" * 60)

print("\nTest complete! You can also translate custom sentences:")
print("  from transformer import translate, make_model, SentencePieceVocab")
print("  # ... load model (see code above)")
print("  result = translate(model, SRC, TGT, 'your sentence here')")

