import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import pickle
import os

class UltraChronoFireTokenizer:
    """Custom tokenizer được train từ dữ liệu Vietnamese conversations"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<eos>': 2,
            '<bos>': 3,
            '<user>': 4,
            '<bot>': 5,
            '<emotion>': 6,
            '<silence>': 7
        }

        # Vietnamese specific patterns
        self.vietnamese_patterns = [
            r'[àáạảãâầấậẩẫăằắặẳẵ]',  # a variants
            r'[èéẹẻẽêềếệểễ]',        # e variants  
            r'[ìíịỉĩ]',              # i variants
            r'[òóọỏõôồốộổỗơờớợởỡ]', # o variants
            r'[ùúụủũưừứựửữ]',        # u variants
            r'[ỳýỵỷỹ]',              # y variants
            r'[đ]'                   # d variant
        ]

        # Emoji patterns for Vietnamese chat
        self.emoji_patterns = [
            r'[😀-🙏]',  # Basic emojis
            r'[🤍💕💖💗💓💙💚💛🧡❤️]',  # Hearts
            r'[🥺🥹😭😢😊😁😂🤣]',  # Common chat emojis
            r'[👏🎉✨💫⭐🌟]'  # Celebration
        ]

        self.bpe_merges = []
        self.bpe_vocab = {}

    def preprocess_text(self, text: str) -> str:
        """Preprocess Vietnamese text for tokenization"""
        # Normalize Unicode
        import unicodedata
        text = unicodedata.normalize('NFC', text)

        # Preserve emojis and special chars
        text = re.sub(r'([.!?])', r' \1 ', text)
        text = re.sub(r'([,;:])', r' \1 ', text)

        # Handle repeated chars (hahaha -> ha ha ha)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Preserve Vietnamese tone marks
        for pattern in self.vietnamese_patterns:
            text = re.sub(f'({pattern})', r'\1', text)

        return text.strip()

    def get_word_frequencies(self, texts: List[str]) -> Counter:
        """Extract word frequencies from training texts"""
        word_freq = Counter()

        for text in texts:
            processed = self.preprocess_text(text.lower())
            words = processed.split()

            for word in words:
                # Split into subwords if too long
                if len(word) > 15:
                    # Split long words into smaller parts
                    for i in range(0, len(word), 5):
                        subword = word[i:i+5]
                        if subword:
                            word_freq[subword] += 1
                else:
                    word_freq[word] += 1

        return word_freq

    def build_bpe_vocab(self, word_freq: Counter, num_merges: int = 10000):
        """Build BPE (Byte Pair Encoding) vocabulary"""

        # Start with character level
        vocab = set()
        for word in word_freq:
            for char in word:
                vocab.add(char)

        # Add special tokens
        for token in self.special_tokens:
            vocab.add(token)

        # Convert words to character sequences
        word_splits = {}
        for word, freq in word_freq.items():
            word_splits[tuple(word)] = freq

        # BPE merging process
        merges = []

        for i in range(num_merges):
            if len(vocab) >= self.vocab_size - len(self.special_tokens):
                break

            # Find most frequent bigram
            bigram_freq = defaultdict(int)

            for word_chars, freq in word_splits.items():
                for j in range(len(word_chars) - 1):
                    bigram = (word_chars[j], word_chars[j+1])
                    bigram_freq[bigram] += freq

            if not bigram_freq:
                break

            best_bigram = max(bigram_freq, key=bigram_freq.get)

            # Merge the best bigram
            merged_token = ''.join(best_bigram)
            vocab.add(merged_token)
            merges.append(best_bigram)

            # Update word splits
            new_word_splits = {}
            for word_chars, freq in word_splits.items():
                new_chars = []
                i = 0
                while i < len(word_chars):
                    if (i < len(word_chars) - 1 and 
                        (word_chars[i], word_chars[i+1]) == best_bigram):
                        new_chars.append(merged_token)
                        i += 2
                    else:
                        new_chars.append(word_chars[i])
                        i += 1
                new_word_splits[tuple(new_chars)] = freq

            word_splits = new_word_splits

        self.bpe_merges = merges
        return vocab

    def train_from_data(self, jsonl_path: str):
        """Train tokenizer from JSONL conversation data"""
        print(f"🚀 Training custom tokenizer from {jsonl_path}")

        # Collect all text
        texts = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                texts.append(data.get('input', ''))
                texts.append(data.get('output', ''))

        print(f"📚 Collected {len(texts)} text samples")

        # Get word frequencies
        word_freq = self.get_word_frequencies(texts)
        print(f"📊 Found {len(word_freq)} unique words")

        # Build BPE vocabulary
        vocab = self.build_bpe_vocab(word_freq)
        vocab_list = sorted(list(vocab))

        # Limit to vocab_size
        if len(vocab_list) > self.vocab_size:
            vocab_list = vocab_list[:self.vocab_size]

        # Build mappings
        self.word_to_id = {}
        self.id_to_word = {}

        # Add special tokens first
        for token, token_id in self.special_tokens.items():
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token

        # Add vocabulary
        for i, word in enumerate(vocab_list):
            if word not in self.special_tokens:
                token_id = len(self.special_tokens) + i
                if token_id < self.vocab_size:
                    self.word_to_id[word] = token_id
                    self.id_to_word[token_id] = word

        print(f"✅ Built vocabulary with {len(self.word_to_id)} tokens")

        # Save tokenizer
        self.save("./model/ultra_tokenizer.pkl")

    def apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word"""
        if not word:
            return []

        chars = list(word)

        for merge in self.bpe_merges:
            new_chars = []
            i = 0
            while i < len(chars):
                if (i < len(chars) - 1 and 
                    chars[i] == merge[0] and chars[i+1] == merge[1]):
                    new_chars.append(''.join(merge))
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars

        return chars

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        processed = self.preprocess_text(text.lower())
        words = processed.split()

        token_ids = [self.special_tokens['<bos>']]

        for word in words:
            # Apply BPE
            subwords = self.apply_bpe(word)

            for subword in subwords:
                if subword in self.word_to_id:
                    token_ids.append(self.word_to_id[subword])
                else:
                    # Handle unknown tokens by character fallback
                    for char in subword:
                        if char in self.word_to_id:
                            token_ids.append(self.word_to_id[char])
                        else:
                            token_ids.append(self.special_tokens['<unk>'])

        token_ids.append(self.special_tokens['<eos>'])
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []

        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]

                if skip_special_tokens and token in self.special_tokens:
                    continue

                tokens.append(token)

        # Join tokens and clean up
        text = ''.join(tokens)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'special_tokens': self.special_tokens,
            'bpe_merges': self.bpe_merges,
            'vocab_size': self.vocab_size
        }

        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)

        print(f"💾 Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)

        tokenizer = cls(tokenizer_data['vocab_size'])
        tokenizer.word_to_id = tokenizer_data['word_to_id']
        tokenizer.id_to_word = tokenizer_data['id_to_word']
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.bpe_merges = tokenizer_data['bpe_merges']

        print(f"📂 Tokenizer loaded from {filepath}")
        return tokenizer

    @property
    def vocab_size_actual(self) -> int:
        return len(self.word_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens['<pad>']

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens['<eos>']

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens['<unk>']

def train_custom_tokenizer(data_path: str = "user_training_data.jsonl", 
                          vocab_size: int = 50000):
    """Train custom tokenizer from training data"""

    if not os.path.exists(data_path):
        print(f"❌ Training data not found: {data_path}")
        return None

    tokenizer = UltraChronoFireTokenizer(vocab_size=vocab_size)
    tokenizer.train_from_data(data_path)

    # Test tokenizer
    test_texts = [
        "Tui buồn quá... 🥺",
        "Haha bạn cute ghê!", 
        "Mình nhớ bạn lắm đó nè~",
        "AI đó, nhưng là phiên bản biết nhớ bạn 😌"
    ]

    print("\n🧪 Testing tokenizer:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded:  {encoded[:10]}...")  # Show first 10 tokens
        print(f"Decoded:  {decoded}")
        print("-" * 50)

    return tokenizer

if __name__ == "__main__":
    tokenizer = train_custom_tokenizer()