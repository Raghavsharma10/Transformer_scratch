def find_tags(tokens, lexicon={}, model=None, morphology=None, context=None, entities=None, default=("NN", "NNP", "CD"), language="en", map=None, **kwargs):
    """ Returns a list of [token, tag]-items for the given list of tokens:
        ["The", "cat", "purs"] => [["The", "DT"], ["cat", "NN"], ["purs", "VB"]]
        Words are tagged using the given lexicon of (word, tag)-items.
        Unknown words are tagged NN by default.
        Unknown words that start with a capital letter are tagged NNP (unless language="de").
        Unknown words that consist only of digits and punctuation marks are tagged CD.
        Unknown words are then improved with morphological rules.
        All words are improved with contextual rules.
        If a model is given, uses model for unknown words instead of morphology and context.
        If map is a function, it is applied to each (token, tag) after applying all rules.
    """
    tagged = []
    # Tag known words.
    for i, token in enumerate(tokens):
        tagged.append([token, lexicon.get(token, i == 0 and lexicon.get(token.lower()) or None)])
    # Tag unknown words.
    for i, (token, tag) in enumerate(tagged):
        prev, next = (None, None), (None, None)
        if i > 0:
            prev = tagged[i-1]
        if i < len(tagged) - 1:
            next = tagged[i+1]
        if tag is None or token in (model is not None and model.unknown or ()):
            # Use language model (i.e., SLP).
            if model is not None:
                tagged[i] = model.apply([token, None], prev, next)
            # Use NNP for capitalized words (except in German).
            elif token.istitle() and language != "de":
                tagged[i] = [token, default[1]]
            # Use CD for digits and numbers.
            elif CD.match(token) is not None:
                tagged[i] = [token, default[2]]
            # Use suffix rules (e.g., -ly = RB).
            elif morphology is not None:
                tagged[i] = morphology.apply([token, default[0]], prev, next)
            # Use suffix rules (English default).
            elif language == "en":
                tagged[i] = _suffix_rules([token, default[0]])
            # Use most frequent tag (NN).
            else:
                tagged[i] = [token, default[0]]
    # Tag words by context.
    if context is not None and model is None:
        tagged = context.apply(tagged)
    # Tag named entities.
    if entities is not None:
        tagged = entities.apply(tagged)
    # Map tags with a custom function.
    if map is not None:
        tagged = [list(map(token, tag)) or [token, default[0]] for token, tag in tagged]
    return tagged