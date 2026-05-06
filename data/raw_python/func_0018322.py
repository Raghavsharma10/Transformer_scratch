def paragraph(node):
    """
    Process a paragraph, which includes all content under it
    """
    text = ''
    if node.string_content is not None:
        text = node.string_content
    o = nodes.paragraph('', ' '.join(text))
    o.line = node.sourcepos[0][0]
    for n in MarkDown(node):
        o.append(n)

    return o