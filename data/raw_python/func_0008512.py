def graphviz_dot(sentence, font="Arial", colors=BLUE):
    """ Returns a dot-formatted string that can be visualized as a graph in GraphViz.
    """
    s  = 'digraph sentence {\n'
    s += '\tranksep=0.75;\n'
    s += '\tnodesep=0.15;\n'
    s += '\tnode [penwidth=1, fontname="%s", shape=record, margin=0.1, height=0.35];\n' % font
    s += '\tedge [penwidth=1];\n'
    s += '\t{ rank=same;\n'
    # Create node groups for words, chunks and PNP chunks.
    for w in sentence.words:
        s += '\t\tword%s [label="<f0>%s|<f1>%s"%s];\n' % (w.index, w.string, w.type, _colorize(w, colors))
    for w in sentence.words[:-1]:
        # Invisible edges forces the words into the right order:
        s += '\t\tword%s -> word%s [color=none];\n' % (w.index, w.index+1)
    s += '\t}\n'
    s += '\t{ rank=same;\n'        
    for i, ch in enumerate(sentence.chunks):
        s += '\t\tchunk%s [label="<f0>%s"%s];\n' % (i+1, "-".join([x for x in (
            ch.type, ch.role, str(ch.relation or '')) if x]) or '-', _colorize(ch, colors))
    for i, ch in enumerate(sentence.chunks[:-1]):
        # Invisible edges forces the chunks into the right order:
        s += '\t\tchunk%s -> chunk%s [color=none];\n' % (i+1, i+2)
    s += '}\n'
    s += '\t{ rank=same;\n'
    for i, ch in enumerate(sentence.pnp):
        s += '\t\tpnp%s [label="<f0>PNP"%s];\n' % (i+1, _colorize(ch, colors))
    s += '\t}\n'
    s += '\t{ rank=same;\n S [shape=circle, margin=0.25, penwidth=2]; }\n'
    # Connect words to chunks.
    # Connect chunks to PNP or S.
    for i, ch in enumerate(sentence.chunks):
        for w in ch:
            s += '\tword%s -> chunk%s;\n' % (w.index, i+1)
        if ch.pnp:
            s += '\tchunk%s -> pnp%s;\n' % (i+1, sentence.pnp.index(ch.pnp)+1)
        else:
            s += '\tchunk%s -> S;\n' % (i+1)
        if ch.type == 'VP':
            # Indicate related chunks with a dotted
            for r in ch.related:
                s += '\tchunk%s -> chunk%s [style=dotted, arrowhead=none];\n' % (
                    i+1, sentence.chunks.index(r)+1)
    # Connect PNP to anchor chunk or S.
    for i, ch in enumerate(sentence.pnp):
        if ch.anchor:
            s += '\tpnp%s -> chunk%s;\n' % (i+1, sentence.chunks.index(ch.anchor)+1)
            s += '\tpnp%s -> S [color=none];\n' % (i+1)
        else:
            s += '\tpnp%s -> S;\n' % (i+1)
    s += "}"
    return s