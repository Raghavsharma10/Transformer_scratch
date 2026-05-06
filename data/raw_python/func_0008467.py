def find_keywords(string, parser, top=10, frequency={}, **kwargs):
    """ Returns a sorted list of keywords in the given string.
        The given parser (e.g., pattern.en.parser) is used to identify noun phrases.
        The given frequency dictionary can be a reference corpus,
        with relative document frequency (df, 0.0-1.0) for each lemma, 
        e.g., {"the": 0.8, "cat": 0.1, ...}
    """
    lemmata = kwargs.pop("lemmata", kwargs.pop("stem", True))
    # Parse the string and extract noun phrases (NP).
    chunks = []
    wordcount = 0
    for sentence in parser.parse(string, chunks=True, lemmata=lemmata).split():
        for w in sentence: # ["cats", "NNS", "I-NP", "O", "cat"]
            if w[2] == "B-NP":
                chunks.append([w])
                wordcount += 1
            elif w[2] == "I-NP" and w[1][:3] == chunks[-1][-1][1][:3] == "NNP":
                chunks[-1][-1][+0] += " " + w[+0] # Collapse NNPs: "Ms Kitty".
                chunks[-1][-1][-1] += " " + w[-1]
            elif w[2] == "I-NP":
                chunks[-1].append(w)
                wordcount += 1
    # Rate the nouns in noun phrases.
    m = {}
    for i, chunk in enumerate(chunks):
        head = True
        if parser.language not in ("ca", "es", "pt", "fr", "it", "pt", "ro"):
            # Head of "cat hair" => "hair".
            # Head of "poils de chat" => "poils".
            chunk = list(reversed(chunk))
        for w in chunk:
            if w[1].startswith("NN"):
                if lemmata:
                    k = w[-1]
                else:
                    k = w[0].lower()
                if not k in m:
                    m[k] = [0.0, set(), 1.0, 1.0, 1.0]
                # Higher score for chunks that appear more frequently.
                m[k][0] += 1 / float(wordcount)
                # Higher score for chunks that appear in more contexts (semantic centrality).
                m[k][1].add(" ".join(map(lambda x: x[0], chunk)).lower())
                # Higher score for chunks at the start (25%) of the text.
                m[k][2] += 1 if float(i) / len(chunks) <= 0.25 else 0
                # Higher score for chunks not in a prepositional phrase.
                m[k][3] += 1 if w[3] == "O" else 0
                # Higher score for chunk head.
                m[k][4] += 1 if head else 0
                head = False
    # Rate tf-idf if a frequency dict is given.
    for k in m:
        if frequency:
            df = frequency.get(k, 0.0)
            df = max(df, 1e-10)
            df = log(1.0 / df, 2.71828)
        else:
            df = 1.0
        m[k][0] = max(1e-10, m[k][0] * df)
        m[k][1] = 1 + float(len(m[k][1]))
    # Sort candidates alphabetically by total score
    # The harmonic mean will emphasize tf-idf score.
    hmean = lambda a: len(a) / sum(1.0 / x for x in a)
    m = [(hmean(m[k]), k) for k in m]
    m = sorted(m, key=lambda x: x[1])
    m = sorted(m, key=lambda x: x[0], reverse=True)
    m = [k for score, k in m]
    return m[:top]