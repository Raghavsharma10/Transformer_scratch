def find_chunks(tagged, language="en"):
    """ The input is a list of [token, tag]-items.
        The output is a list of [token, tag, chunk]-items:
        The/DT nice/JJ fish/NN is/VBZ dead/JJ ./. =>
        The/DT/B-NP nice/JJ/I-NP fish/NN/I-NP is/VBZ/B-VP dead/JJ/B-ADJP ././O
    """
    chunked = [x for x in tagged]
    tags = "".join("%s%s" % (tag, SEPARATOR) for token, tag in tagged)
    # Use Germanic or Romance chunking rules according to given language.
    for tag, rule in CHUNKS[int(language in ("ca", "es", "pt", "fr", "it", "pt", "ro"))]:
        for m in rule.finditer(tags):
            # Find the start of chunks inside the tags-string.
            # Number of preceding separators = number of preceding tokens.
            i = m.start()
            j = tags[:i].count(SEPARATOR)
            n = m.group(0).count(SEPARATOR)
            for k in range(j, j+n):
                if len(chunked[k]) == 3:
                    continue
                if len(chunked[k]) < 3:
                    # A conjunction or comma cannot be start of a chunk.
                    if k == j and chunked[k][1] in ("CC", "CJ", ","):
                        j += 1
                    # Mark first token in chunk with B-.
                    elif k == j:
                        chunked[k].append("B-" + tag)
                    # Mark other tokens in chunk with I-.
                    else:
                        chunked[k].append("I-" + tag)
    # Mark chinks (tokens outside of a chunk) with O-.
    for chink in filter(lambda x: len(x) < 3, chunked):
        chink.append("O")
    # Post-processing corrections.
    for i, (word, tag, chunk) in enumerate(chunked):
        if tag.startswith("RB") and chunk == "B-NP":
            # "Perhaps you" => ADVP + NP
            # "Really nice work" => NP
            # "Really, nice work" => ADVP + O + NP
            if i < len(chunked)-1 and not chunked[i+1][1].startswith("JJ"):
                chunked[i+0][2] = "B-ADVP"
                chunked[i+1][2] = "B-NP"
            if i < len(chunked)-1 and chunked[i+1][1] in ("CC", "CJ", ","):
                chunked[i+1][2] = "O"
            if i < len(chunked)-2 and chunked[i+1][2] == "O":
                chunked[i+2][2] = "B-NP"
    return chunked