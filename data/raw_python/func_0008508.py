def _parse_tokens(chunk, format=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
    """ Parses tokens from <word> elements in the given XML <chunk> element.
        Returns a flat list of tokens, in which each token is [WORD, POS, CHUNK, PNP, RELATION, ANCHOR, LEMMA].
        If a <chunk type="PNP"> is encountered, traverses all of the chunks in the PNP.
    """
    tokens = []
    # Only process <chunk> and <chink> elements, 
    # text nodes in between return an empty list.
    if not (chunk.tag == XML_CHUNK or chunk.tag == XML_CHINK):
        return []
    type = chunk.get(XML_TYPE, "O")
    if type == "PNP":
        # For, <chunk type="PNP">, recurse all the child chunks inside the PNP.
        for ch in chunk:
            tokens.extend(_parse_tokens(ch, format))
        # Tag each of them as part of the PNP.
        if PNP in format:
            i = format.index(PNP)
            for j, token in enumerate(tokens):
                token[i] = (j==0 and "B-" or "I-") + "PNP"
        # Store attachments so we can construct anchor id's in parse_string().
        # This has to be done at the end, when all the chunks have been found.
        a = chunk.get(XML_OF).split(_UID_SEPARATOR)[-1]
        if a:
            _attachments.setdefault(a, [])
            _attachments[a].append(tokens)
        return tokens
    # For <chunk type-"VP" id="1">, the relation is VP-1.
    # For <chunk type="NP" relation="OBJ" of="1">, the relation is NP-OBJ-1.
    relation = _parse_relation(chunk, type)
    # Process all of the <word> elements in the chunk, for example:
    # <word type="NN" lemma="pizza">pizza</word> => [pizza, NN, I-NP, O, NP-OBJ-1, O, pizza]
    for word in filter(lambda n: n.tag == XML_WORD, chunk):
        tokens.append(_parse_token(word, chunk=type, relation=relation, format=format))
    # Add the IOB chunk tags:
    # words at the start of a chunk are marked with B-, words inside with I-.
    if CHUNK in format:
        i = format.index(CHUNK)
        for j, token in enumerate(tokens):
            token[i] = token[i] != "O" and ((j==0 and "B-" or "I-") + token[i]) or "O"
    # The chunk can be the anchor of one or more PNP chunks.
    # Store anchors so we can construct anchor id's in parse_string().
    a = chunk.get(XML_ANCHOR, "").split(_UID_SEPARATOR)[-1]
    if a: 
        _anchors[a] = tokens
    return tokens