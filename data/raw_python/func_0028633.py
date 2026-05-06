def absolute(parser, token):
    '''
    Returns a full absolute URL based on the request host.

    This template tag takes exactly the same paramters as url template tag.
    '''
    node = url(parser, token)
    return AbsoluteUrlNode(
        view_name=node.view_name,
        args=node.args,
        kwargs=node.kwargs,
        asvar=node.asvar
    )