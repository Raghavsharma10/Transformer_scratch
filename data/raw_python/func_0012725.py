def tag_and_stem(text):
    """
    Returns a list of (stem, tag, token) triples:

    - stem: the word's uninflected form
    - tag: the word's part of speech
    - token: the original word, so we can reconstruct it later
    """
    tokens = tokenize(text)
    tagged = nltk.pos_tag(tokens)
    out = []
    for token, tag in tagged:
        stem = morphy_stem(token, tag)
        out.append((stem, tag, token))
    return out