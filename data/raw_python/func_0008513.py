def table(sentence, fill=1, placeholder="-"):
    """ Returns a string where the tags of tokens in the sentence are organized in outlined columns.
    """
    tags  = [WORD, POS, IOB, CHUNK, ROLE, REL, PNP, ANCHOR, LEMMA]
    tags += [tag for tag in sentence.token if tag not in tags]
    def format(token, tag):
        # Returns the token tag as a string.
        if   tag == WORD   : s = token.string
        elif tag == POS    : s = token.type
        elif tag == IOB    : s = token.chunk and (token.index == token.chunk.start and "B" or "I")
        elif tag == CHUNK  : s = token.chunk and token.chunk.type
        elif tag == ROLE   : s = token.chunk and token.chunk.role
        elif tag == REL    : s = token.chunk and token.chunk.relation and str(token.chunk.relation)
        elif tag == PNP    : s = token.chunk and token.chunk.pnp and token.chunk.pnp.type
        elif tag == ANCHOR : s = token.chunk and token.chunk.anchor_id
        elif tag == LEMMA  : s = token.lemma
        else               : s = token.custom_tags.get(tag)
        return s or placeholder
    def outline(column, fill=1, padding=3, align="left"):
        # Add spaces to each string in the column so they line out to the highest width.
        n = max([len(x) for x in column]+[fill])
        if align == "left"  : return [x+" "*(n-len(x))+" "*padding for x in column]
        if align == "right" : return [" "*(n-len(x))+x+" "*padding for x in column]
    
    # Gather the tags of the tokens in the sentece per column.
    # If the IOB-tag is I-, mark the chunk tag with "^".
    # Add the tag names as headers in each column.
    columns = [[format(token, tag) for token in sentence] for tag in tags]
    columns[3] = [columns[3][i]+(iob == "I" and " ^" or "") for i, iob in enumerate(columns[2])]
    del columns[2]
    for i, header in enumerate(['word', 'tag', 'chunk', 'role', 'id', 'pnp', 'anchor', 'lemma']+tags[9:]):
        columns[i].insert(0, "")
        columns[i].insert(0, header.upper())
    # The left column (the word itself) is outlined to the right,
    # and has extra spacing so that words across sentences line out nicely below each other.
    for i, column in enumerate(columns):
        columns[i] = outline(column, fill+10*(i==0), align=("left","right")[i==0])
    # Anchor column is useful in MBSP but not in pattern.en.
    if not MBSP:
        del columns[6] 
    # Create a string with one row (i.e., one token) per line.
    return "\n".join(["".join([x[i] for x in columns]) for i in range(len(columns[0]))])