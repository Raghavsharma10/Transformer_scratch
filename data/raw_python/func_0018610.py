def tree_text(node):
    """
    >>> tree_text(parse_minidom('<h1>one</h1>two<div>three<em>four</em></div>'))
    'one two three four'
    """
    text = []
    for descendant in walk_dom(node):
        if is_text(descendant):
            text.append(descendant.nodeValue)
    return ' '.join(text)