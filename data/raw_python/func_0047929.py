def ast_to_headings(node):
    """
    Walks AST and returns a list of headings
    """

    Heading = namedtuple('Heading', ['level', 'title'])

    level = None
    walker = node.walker()
    headings = []

    event = walker.nxt()
    while event is not None:
        entering = event['entering']
        node = event['node']

        if node.t == 'Heading':
            if entering:
                level = node.level
            else:
                level = None
        elif level:
            if node.t != 'Text':
                raise Exception('Unexpected node {}, only text may be within a heading.'.format(node.t))

            headings.append(Heading(level=level, title=node.literal))

        event = walker.nxt()

    return headings