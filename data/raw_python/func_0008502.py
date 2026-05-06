def chunked(sentence):
    """ Returns a list of Chunk and Chink objects from the given sentence.
        Chink is a subclass of Chunk used for words that have Word.chunk == None
        (e.g., punctuation marks, conjunctions).
    """
    # For example, to construct a training vector with the head of previous chunks as a feature.
    # Doing this with Sentence.chunks would discard the punctuation marks and conjunctions
    # (Sentence.chunks only yields Chunk objects), which amy be useful features.
    chunks = []
    for word in sentence:
        if word.chunk is not None:
            if len(chunks) == 0 or chunks[-1] != word.chunk:
                chunks.append(word.chunk)
        else:
            ch = Chink(sentence)
            ch.append(word.copy(ch))
            chunks.append(ch)
    return chunks