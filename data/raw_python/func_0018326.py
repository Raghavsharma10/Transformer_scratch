def literal(node):
    """
    Inline code
    """
    rendered = []
    try:
        if node.info is not None:
            l = Lexer(node.literal, node.info, tokennames="long")
            for _ in l:
                rendered.append(node.inline(classes=_[0], text=_[1]))
    except:
        pass

    classes = ['code']
    if node.info is not None:
        classes.append(node.info)
    if len(rendered) > 0:
        o = nodes.literal(classes=classes)
        for element in rendered:
            o += element
    else:
        o = nodes.literal(text=node.literal, classes=classes)

    for n in MarkDown(node):
        o += n
    return o