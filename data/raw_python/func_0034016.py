def _get_node(template, context, name):
    '''
    taken originally from
    http://stackoverflow.com/questions/2687173/django-how-can-i-get-a-block-from-a-template
    '''
    for node in template:
        if isinstance(node, BlockNode) and node.name == name:
            return node.nodelist.render(context)
        elif isinstance(node, ExtendsNode):
            return _get_node(node.nodelist, context, name)

    # raise Exception("Node '%s' could not be found in template." % name)
    return ""