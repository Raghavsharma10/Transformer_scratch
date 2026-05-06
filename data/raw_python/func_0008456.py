def ngrams(string, n=3, punctuation=PUNCTUATION, continuous=False):
    """ Returns a list of n-grams (tuples of n successive words) from the given string.
        Alternatively, you can supply a Text or Sentence object.
        With continuous=False, n-grams will not run over sentence markers (i.e., .!?).
        Punctuation marks are stripped from words.
    """
    def strip_punctuation(s, punctuation=set(punctuation)):
        return [w for w in s if (isinstance(w, Word) and w.string or w) not in punctuation]
    if n <= 0:
        return []
    if isinstance(string, basestring):
        s = [strip_punctuation(s.split(" ")) for s in tokenize(string)]
    if isinstance(string, Sentence):
        s = [strip_punctuation(string)]
    if isinstance(string, Text):
        s = [strip_punctuation(s) for s in string]
    if continuous:
        s = [sum(s, [])]
    g = []
    for s in s:
        #s = [None] + s + [None]
        g.extend([tuple(s[i:i+n]) for i in range(len(s)-n+1)])
    return g