def build_embedding_weights(word_index, embeddings_index):
    """Builds an embedding matrix for all words in vocab using embeddings_index
    """
    logger.info('Loading embeddings for all words in the corpus')
    embedding_dim = list(embeddings_index.values())[0].shape[-1]

    # setting special tokens such as UNK and PAD to 0
    # all other words are also set to 0.
    embedding_weights = np.zeros((len(word_index), embedding_dim))

    for word, i in word_index.items():
        word_vector = embeddings_index.get(word)
        if word_vector is not None:
            embedding_weights[i] = word_vector

    return embedding_weights