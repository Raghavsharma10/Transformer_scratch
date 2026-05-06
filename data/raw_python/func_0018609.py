def tree_words(node):
    """Return all the significant text below the given node as a list of words.
    >>> list(tree_words(parse_minidom('<h1>one</h1> two <div>three<em>four</em></div>')))
    ['one', 'two', 'three', 'four']
    """
    for word in split_text(tree_text(node)):
        word = word.strip()
        if word:
            yield word