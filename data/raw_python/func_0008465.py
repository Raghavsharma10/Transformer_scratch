def find_prepositions(chunked):
    """ The input is a list of [token, tag, chunk]-items.
        The output is a list of [token, tag, chunk, preposition]-items.
        PP-chunks followed by NP-chunks make up a PNP-chunk.
    """
    # Tokens that are not part of a preposition just get the O-tag.
    for ch in chunked:
        ch.append("O")
    for i, chunk in enumerate(chunked):
        if chunk[2].endswith("PP") and chunk[-1] == "O":
            # Find PP followed by other PP, NP with nouns and pronouns, VP with a gerund.
            if i < len(chunked)-1 and \
             (chunked[i+1][2].endswith(("NP", "PP")) or \
              chunked[i+1][1] in ("VBG", "VBN")):
                chunk[-1] = "B-PNP"
                pp = True
                for ch in chunked[i+1:]:
                    if not (ch[2].endswith(("NP", "PP")) or ch[1] in ("VBG", "VBN")):
                        break
                    if ch[2].endswith("PP") and pp:
                        ch[-1] = "I-PNP"
                    if not ch[2].endswith("PP"):
                        ch[-1] = "I-PNP"
                        pp = False
    return chunked