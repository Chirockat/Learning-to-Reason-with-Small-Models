import numpy as np
import re


class DataLoader:
    def __init__(self, text, window_size=2):
        self.window_size = window_size


        self.tokens = self._clean_and_tokenize(text)


        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

        self._build_vocab()

    def _clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def _build_vocab(self):
        unique_words = list(set(self.tokens))
        self.vocab_size = len(unique_words)

        for i, word in enumerate(unique_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        print(f"Unique Words: {self.vocab_size}")
        print(f"All Words: {len(self.tokens)}")

    def get_training_pairs(self):

        pairs = []


        token_ids = [self.word_to_id[word] for word in self.tokens]

        for i, center_word_id in enumerate(token_ids):
            start = max(0, i - self.window_size)
            end = min(len(token_ids), i + self.window_size + 1)


            for j in range(start, end):
                if i != j:
                    context_word_id = token_ids[j]
                    pairs.append((center_word_id, context_word_id))

        return np.array(pairs)


# test
if __name__ == "__main__":
    sample_text = "Mały pies gonił czarnego kota za płotem"


    loader = DataLoader(sample_text, window_size=1)


    training_data = loader.get_training_pairs()

    print("\nPrzykładowe 5 par uczących (jako liczby):")
    print(training_data[:5])

    print("\nJak to wygląda po przetłumaczeniu z powrotem na słowa:")
    for center_id, context_id in training_data[:5]:
        center_word = loader.id_to_word[center_id]
        context_word = loader.id_to_word[context_id]
        print(f"Środek: '{center_word}' -> Kontekst: '{context_word}'")