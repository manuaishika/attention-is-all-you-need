"""
Test the trained Transformer model for translation
"""

import torch
from transformer import make_model, translate, SentencePieceVocab

# Fix Unicode encoding for Windows console
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7 fallback
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("Testing Trained Transformer Model")
print("=" * 70)

# Load the trained model
print("\nLoading model...")
checkpoint = torch.load('transformer_model.pth', map_location=device)

# Create model with same architecture
model = make_model(
    checkpoint['src_vocab_size'],
    checkpoint['tgt_vocab_size'],
    N=checkpoint['N'],
    d_model=checkpoint['d_model'],
    d_ff=checkpoint['d_ff'],
    h=checkpoint['h'],
    dropout=checkpoint['dropout']
).to(device)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("OK: Model loaded successfully!")

# Load tokenizers
print("\nLoading tokenizers...")
SRC = SentencePieceVocab("spm_model.model")
TGT = SentencePieceVocab("spm_model.model")
print("OK: Tokenizers loaded!")

# Test translations
print("\n" + "=" * 70)
print("Translation Tests")
print("=" * 70)

test_sentences = [
    "hello how are you",
    "what is your name",
    "i am a student",
    "thank you very much",
    "good morning",
    "where do you live",
    "i love programming"
]

print("\nTranslating sentences...\n")
for i, sent in enumerate(test_sentences, 1):
    translation = translate(model, SRC, TGT, sent)
    print(f"{i}. English: {sent}")
    print(f"   Translation: {translation}")
    print()

# Interactive mode
print("=" * 70)
print("Interactive Translation Mode")
print("=" * 70)
print("Enter English sentences to translate (type 'quit' to exit):\n")

while True:
    try:
        sentence = input("English: ").strip()
        if sentence.lower() in ['quit', 'exit', 'q']:
            break
        if not sentence:
            continue
        
        translation = translate(model, SRC, TGT, sentence)
        print(f"Translation: {translation}\n")
    except KeyboardInterrupt:
        print("\n\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}\n")

print("\nDone!")

