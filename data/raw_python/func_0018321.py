def customWalker(node, space=''):
    """
    A convenience function to ease debugging. It will print the node structure that's returned from CommonMark

    The usage would be something like:

    >>> content = Parser().parse('Some big text block\n===================\n\nwith content\n')
    >>> customWalker(content)
    document
        heading
            text	Some big text block
        paragraph
            text	with content

    Spaces are used to convey nesting
    """
    txt = ''
    try:
        txt = node.literal
    except:
        pass

    if txt is None or txt == '':
        print('{}{}'.format(space, node.t))
    else:
        print('{}{}\t{}'.format(space, node.t, txt))

    cur = node.first_child
    if cur:
        while cur is not None:
            customWalker(cur, space + '    ')
            cur = cur.nxt