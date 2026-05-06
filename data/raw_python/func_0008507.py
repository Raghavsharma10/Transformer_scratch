def parse_string(xml):
    """ Returns a slash-formatted string from the given XML representation.
        The return value is a TokenString (for MBSP) or TaggedString (for Pattern).
    """
    string = ""
    # Traverse all the <sentence> elements in the XML.
    dom = XML(xml)
    for sentence in dom(XML_SENTENCE):
        _anchors.clear()     # Populated by calling _parse_tokens().
        _attachments.clear() # Populated by calling _parse_tokens().
        # Parse the language from <sentence language="">.
        language = sentence.get(XML_LANGUAGE, "en")
        # Parse the token tag format from <sentence token="">.
        # This information is returned in TokenString.tags,
        # so the format and order of the token tags is retained when exporting/importing as XML.
        format = sentence.get(XML_TOKEN, [WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA])
        format = not isinstance(format, basestring) and format or format.replace(" ","").split(",")
        # Traverse all <chunk> and <chink> elements in the sentence.
        # Find the <word> elements inside and create tokens.
        tokens = []
        for chunk in sentence:
            tokens.extend(_parse_tokens(chunk, format))
        # Attach PNP's to their anchors.
        # Keys in _anchors have linked anchor chunks (each chunk is a list of tokens).
        # The keys correspond to the keys in _attachments, which have linked PNP chunks.
        if ANCHOR in format:
            A, P, a, i = _anchors, _attachments, 1, format.index(ANCHOR)
            for id in sorted(A.keys()):
                for token in A[id]:
                    token[i] += "-"+"-".join(["A"+str(a+p) for p in range(len(P[id]))])
                    token[i]  = token[i].strip("O-")
                for p, pnp in enumerate(P[id]):
                    for token in pnp: 
                        token[i] += "-"+"P"+str(a+p)
                        token[i]  = token[i].strip("O-")
                a += len(P[id])
        # Collapse the tokens to string.
        # Separate multiple sentences with a new line.
        tokens = ["/".join([tag for tag in token]) for token in tokens]
        tokens = " ".join(tokens)
        string += tokens + "\n"
    # Return a TokenString, which is a unicode string that transforms easily
    # into a plain str, a list of tokens, or a Sentence.
    try:
        if MBSP: from mbsp import TokenString
        return TokenString(string.strip(), tags=format, language=language)
    except:
        return TaggedString(string.strip(), tags=format, language=language)