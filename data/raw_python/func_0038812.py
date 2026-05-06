def make_pagination_headers(request, limit, curpage, total, links=False):
    """Return Link Hypermedia Header."""
    lastpage = math.ceil(total / limit) - 1
    headers = {'X-Total-Count': str(total), 'X-Limit': str(limit),
               'X-Page-Last': str(lastpage), 'X-Page': str(curpage)}
    if links:
        base = "{}?%s".format(request.path)
        links = {}
        links['first'] = base % urlencode(dict(request.query, **{VAR_PAGE: 0}))
        links['last'] = base % urlencode(dict(request.query, **{VAR_PAGE: lastpage}))
        if curpage:
            links['prev'] = base % urlencode(dict(request.query, **{VAR_PAGE: curpage - 1}))
        if curpage < lastpage:
            links['next'] = base % urlencode(dict(request.query, **{VAR_PAGE: curpage + 1}))
        headers['Link'] = ",".join(['<%s>; rel="%s"' % (v, n) for n, v in links.items()])
    return headers