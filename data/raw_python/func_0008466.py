def find_relations(chunked):
    """ The input is a list of [token, tag, chunk]-items.
        The output is a list of [token, tag, chunk, relation]-items.
        A noun phrase preceding a verb phrase is perceived as sentence subject.
        A noun phrase following a verb phrase is perceived as sentence object.
    """
    tag = lambda token: token[2].split("-")[-1] # B-NP => NP
    # Group successive tokens with the same chunk-tag.
    chunks = []
    for token in chunked:
        if len(chunks) == 0 \
        or token[2].startswith("B-") \
        or tag(token) != tag(chunks[-1][-1]):
            chunks.append([])
        chunks[-1].append(token+["O"])
    # If a VP is preceded by a NP, the NP is tagged as NP-SBJ-(id).
    # If a VP is followed by a NP, the NP is tagged as NP-OBJ-(id).
    # Chunks that are not part of a relation get an O-tag.
    id = 0
    for i, chunk in enumerate(chunks):
        if tag(chunk[-1]) == "VP" and i > 0 and tag(chunks[i-1][-1]) == "NP":
            if chunk[-1][-1] == "O":
                id += 1
            for token in chunk:
                token[-1] = "VP-" + str(id)
            for token in chunks[i-1]:
                token[-1] += "*NP-SBJ-" + str(id)
                token[-1] = token[-1].lstrip("O-*")
        if tag(chunk[-1]) == "VP" and i < len(chunks)-1 and tag(chunks[i+1][-1]) == "NP":
            if chunk[-1][-1] == "O":
                id += 1
            for token in chunk:
                token[-1] = "VP-" + str(id)
            for token in chunks[i+1]:
                token[-1] = "*NP-OBJ-" + str(id)
                token[-1] = token[-1].lstrip("O-*")
    # This is more a proof-of-concept than useful in practice:
    # PP-LOC = be + in|at + the|my
    # PP-DIR = go + to|towards + the|my
    for i, chunk in enumerate(chunks):
        if 0 < i < len(chunks)-1 and len(chunk) == 1 and chunk[-1][-1] == "O":
            t0, t1, t2 = chunks[i-1][-1], chunks[i][0], chunks[i+1][0] # previous / current / next
            if tag(t1) == "PP" and t2[1] in ("DT", "PR", "PRP$"):
                if t0[0] in BE and t1[0] in ("in", "at")      : t1[-1] = "PP-LOC"
                if t0[0] in GO and t1[0] in ("to", "towards") : t1[-1] = "PP-DIR"
    related = []; [related.extend(chunk) for chunk in chunks]
    return related