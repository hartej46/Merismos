# BPE Tokenizer:

import regex as re

class MinBPE:
    def __init__ (self):

        self.vocab = {}

        self.merges = {}

    def train(self,text,vocab_size):

        split_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = re.findall(re.compile(split_pattern),text)

        token_sequences = [list(word.encode("utf-8")) for word in words ]

        self.vocab = {i : bytes([i]) for i in range(256)}

        
        num_merges = vocab_size - 256
        current_idx = 256

        for _ in range(num_merges) :

            global_stats = {}

            for seq in token_sequences:

                stats = self.get_stats(seq)

                for pair,count in stats.items():
                    global_stats[pair] = global_stats.get(pair,0) + count

            if not global_stats:
                break

            best_pair = max(global_stats, key = global_stats.get)

            self.merges[best_pair] = current_idx

            self.vocab[current_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            token_sequences = [self.merge(seq, best_pair, current_idx) for seq in token_sequences]

            current_idx += 1

            print(f"{_}")

    def get_stats(self, ids) :

        counts = {}

        for pair in zip(ids,ids[1:]):

            counts[pair] = counts.get(pair,0) + 1

        return counts
    
    def merge(self,ids,pair,idx):

        new_ids = []
        i = 0

        while i< len(ids):

            if i <len(ids) - 1 and (ids[i],ids[i+1]) == pair :
                new_ids.append(idx)
                i +=2

            else:
                new_ids.append(ids[i])
                i +=1
        return new_ids
    
    def encode(self,text):

        split_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = re.findall(re.compile(split_pattern), text)

        final_ids = []

        for word in words:

            ids = list(word.encode("utf-8"))

            while len(ids) >=2 :

                stats = self.get_stats(ids)

                pair = min(stats.keys(), key = lambda p : self.merges.get(p , float("inf")))

                if pair not in self.merges:
                    break

                new_idx = self.merges[pair]

                ids = self.merge(ids, pair, new_idx)

            final_ids.extend(ids)

        return final_ids
    
    def decode(self,ids):
        
        byte_segments = [self.vocab.get(idx,b'') for idx in ids]

        byte_string = b''.join(byte_segments)

        return byte_string.decode("utf-8", errors = "replace")
        

