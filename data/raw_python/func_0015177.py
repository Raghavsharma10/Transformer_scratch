def format_search(q, **kwargs):
    '''Formats the results of a search'''
    m = search(q, **kwargs)
    count = m['count']
    if not count:
        raise DapiCommError('Could not find any DAP packages for your query.')
        return
    for mdap in m['results']:
        mdap = mdap['content_object']
        return _format_dap_with_description(mdap)