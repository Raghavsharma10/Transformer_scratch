def site(parser, token):
    '''
    Returns a full absolute URL based on the current site.

    This template tag takes exactly the same paramters as url template tag.
    '''
    node = url(parser, token)
    return SiteUrlNode(
        view_name=node.view_name,
        args=node.args,
        kwargs=node.kwargs,
        asvar=node.asvar
    )