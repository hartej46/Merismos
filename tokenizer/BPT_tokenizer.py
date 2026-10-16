import json
from collections import Counter
import regex as re
from tqdm import trange


class MinBPE:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.split_pattern = (
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.compiled_pattern = re.compile(self.split_pattern)

    def train(self, text: str, vocab_size: int):
        words = self.compiled_pattern.findall(text)

        # 1. Deduplicate: count frequency of each unique byte sequence
        word_freqs = Counter(tuple(w.encode("utf-8")) for w in words)

        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size - 256
        current_idx = 256

        # 2. Pre-compute initial pair statistics weighted by word frequencies
        pair_counts = Counter()
        for word_tuple, freq in word_freqs.items():
            for pair in zip(word_tuple, word_tuple[1:]):
                pair_counts[pair] += freq

        for _ in trange(num_merges, desc="Training BPE"):
            if not pair_counts:
                break

            # Find the most frequent pair across the whole corpus
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] == 0:
                break

            self.merges[best_pair] = current_idx
            self.vocab[current_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # 3. Only process and update words containing the best_pair
            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                if best_pair[0] not in word_tuple:
                    new_word_freqs[word_tuple] = freq
                    continue

                # Merge the pair inside this word
                merged_word = self._merge_tuple(word_tuple, best_pair, current_idx)
                new_word_freqs[merged_word] = freq

                # If the word changed, update pair_counts incrementally
                if merged_word != word_tuple:
                    for old_pair in zip(word_tuple, word_tuple[1:]):
                        pair_counts[old_pair] -= freq
                    for new_pair in zip(merged_word, merged_word[1:]):
                        pair_counts[new_pair] += freq

            word_freqs = new_word_freqs
            # Remove the pair we just merged
            del pair_counts[best_pair]
            current_idx += 1

    def _merge_tuple(self, ids: tuple, pair: tuple, idx: int) -> tuple:
        """Fast merge for a single tuple of token IDs."""
        new_ids = []
        i = 0
        p0, p1 = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return tuple(new_ids)

    def encode(self, text: str) -> list[int]:
        """Encodes text using memoized word-level lookups."""
        words = self.compiled_pattern.findall(text)
        final_ids = []
        cache = {}

        for word in words:
            if word in cache:
                final_ids.extend(cache[word])
                continue

            ids = tuple(word.encode("utf-8"))
            while len(ids) >= 2:
                # Find all consecutive pairs in the current word
                pairs = zip(ids, ids[1:])
                # Find the pair with the lowest merge rank (trained earliest)
                pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break

                ids = self._merge_tuple(ids, pair, self.merges[pair])

            cache[word] = ids
            final_ids.extend(ids)

        return final_ids

    def decode(self, ids: list[int]) -> str:
        byte_segments = [self.vocab.get(idx, b"") for idx in ids]
        return b"".join(byte_segments).decode("utf-8", errors="replace")

    def save(self, file_path: str):
        serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(self.merges)} merges to {file_path}")

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {}
        for k, v in data.items():
            p1, p2 = k.split(",")
            self.merges[(int(p1), int(p2))] = int(v)

        self.vocab = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for pair, idx in sorted_merges:
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        print(f"Loaded {len(self.merges)} merges. Vocab size: {len(self.vocab)}")
