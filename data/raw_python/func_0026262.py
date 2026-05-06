def word_tokenize(text, stopwords=_stopwords, ngrams=None, min_length=0, ignore_numeric=True):
    """
    Parses the given text and yields tokens which represent words within
    the given text. Tokens are assumed to be divided by any form of
    whitespace character.
    """
    if ngrams is None:
        ngrams = 1

    text = re.sub(re.compile('\'s'), '', text)  # Simple heuristic
    text = re.sub(_re_punctuation, '', text)

    matched_tokens = re.findall(_re_token, text.lower())
    for tokens in get_ngrams(matched_tokens, ngrams):
        for i in range(len(tokens)):
            tokens[i] = tokens[i].strip(punctuation)

            if len(tokens[i]) < min_length or tokens[i] in stopwords:
                break
            if ignore_numeric and isnumeric(tokens[i]):
                break
        else:
            yield tuple(tokens)