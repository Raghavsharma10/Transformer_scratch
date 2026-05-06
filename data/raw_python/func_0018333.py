def listNode(node):
    """
    A list (numbered or not)
    For numbered lists, the suffix is only rendered as . in html
    """
    if node.list_data['type'] == u'bullet':
        o = nodes.bullet_list(bullet=node.list_data['bullet_char'])
    else:
        o = nodes.enumerated_list(suffix=node.list_data['delimiter'], enumtype='arabic', start=node.list_data['start'])
    for n in MarkDown(node):
        o += n
    return o