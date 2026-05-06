def nltk_tree(sentence):
    """ Returns an NLTK nltk.tree.Tree object from the given Sentence.
        The NLTK module should be on the search path somewhere.
    """
    from nltk import tree
    def do_pnp(pnp):
        # Returns the PNPChunk (and the contained Chunk objects) in NLTK bracket format.
        s = ' '.join([do_chunk(ch) for ch in pnp.chunks])
        return '(PNP %s)' % s
    
    def do_chunk(ch):
        # Returns the Chunk in NLTK bracket format. Recurse attached PNP's.
        s = ' '.join(['(%s %s)' % (w.pos, w.string) for w in ch.words])
        s+= ' '.join([do_pnp(pnp) for pnp in ch.attachments])
        return '(%s %s)' % (ch.type, s)
    
    T = ['(S']
    v = [] # PNP's already visited.
    for ch in sentence.chunked():
        if not ch.pnp and isinstance(ch, Chink):
            T.append('(%s %s)' % (ch.words[0].pos, ch.words[0].string))
        elif not ch.pnp:
            T.append(do_chunk(ch))
        #elif ch.pnp not in v:
        elif ch.pnp.anchor is None and ch.pnp not in v:
            # The chunk is part of a PNP without an anchor.
            T.append(do_pnp(ch.pnp))
            v.append(ch.pnp)
    T.append(')')
    return tree.bracket_parse(' '.join(T))