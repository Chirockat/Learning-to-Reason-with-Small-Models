import numpy as np
from data_loader import DataLoader
from model import Word2VecModel
from tqdm import tqdm

CORPUS_PATH = 'data/corpus.txt'
MAX_WORDS = 15000
TEST_WORDS = ["death", "wolf", "king"]
TOP_N_SIMILAR = 2

WINDOW_SIZE = 3
MIN_COUNT = 3
EMBEDDING_DIM = 50
LEARNING_RATE = 0.05
EPOCHS = 5


def get_similar_words(target_word, word_vectors, word_to_id, id_to_word, top_n=3):
    if target_word not in word_to_id:
        return f"Word '{target_word}' not found in vocabulary."

    target_id = word_to_id[target_word]
    target_vector = word_vectors[target_id]

    dot_products = np.dot(word_vectors, target_vector)

    norms_all = np.linalg.norm(word_vectors, axis=1)
    norm_target = np.linalg.norm(target_vector)

    norms_all[norms_all == 0] = 1e-9
    norm_target = norm_target if norm_target != 0 else 1e-9

    similarities = dot_products / (norms_all * norm_target)
    sorted_indices = np.argsort(similarities)[::-1]

    similar_words = []
    for idx in sorted_indices:
        if idx != target_id:
            word = id_to_word[idx]
            sim_score = similarities[idx]
            similar_words.append((word, sim_score))
            if len(similar_words) == top_n:
                break

    return similar_words

def train():
    full_text = open(CORPUS_PATH, 'r', encoding='utf-8').read()
    text_subset = " ".join(full_text.split()[:MAX_WORDS])

    print(f"Selected a subset of {MAX_WORDS} words for testing.")

    loader = DataLoader(text_subset, window_size=WINDOW_SIZE, min_count=MIN_COUNT)
    training_data = loader.get_training_pairs()

    print("\nInitializing Word2Vec model...")
    model = Word2VecModel(vocab_size=loader.vocab_size, embedding_dim=EMBEDDING_DIM, learning_rate=LEARNING_RATE)

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        total_loss = 0

        pbar = tqdm(enumerate(training_data), total=len(training_data),
                    desc=f"Epoch {epoch + 1}/{EPOCHS}", mininterval=0.5)

        for i, (center_word_id, context_word_id) in pbar:
            model.forward(center_word_id)
            loss = model.backward(center_word_id, context_word_id)
            total_loss += loss

            if i % 1000 == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    print("\nTraining finished!")
    word_vectors = model.W1

    print("\n--- WORD SIMILARITY TEST ---")
    for word in TEST_WORDS:
        clean_word = loader._clean_and_tokenize(word)
        if clean_word:
            search_word = clean_word[0]
            results = get_similar_words(search_word, word_vectors, loader.word_to_id, loader.id_to_word, top_n=TOP_N_SIMILAR)

            print(f"\nClosest words to '{search_word}':")
            if isinstance(results, str):
                print(results)
            else:
                for sim_word, score in results:
                    print(f" -> {sim_word} (similarity: {score:.4f})")

if __name__ == "__main__":
    train()