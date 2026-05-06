def range_return(request, items):
    """
    Determine what range of objects to return.

    Will check fot both `Range` and `X-Range` headers in the request and
    set both `Content-Range` and 'X-Content-Range' headers.

    :rtype: list
    """
    if ('Range' in request.headers):
        range = parse_range_header(request.headers['Range'])
    elif 'X-Range' in request.headers:
        range = parse_range_header(request.headers['X-Range'])
    else:
        range = {
            'start': 0,
            'finish': MAX_NUMBER_ITEMS - 1,
            'count': MAX_NUMBER_ITEMS
        }
    filtered = items[range['start']:range['finish'] + 1]
    if len(filtered) < range['count']:
        # Something was stripped, deal with it
        range['count'] = len(filtered)
        range['finish'] = range['start'] + range['count'] - 1
    if range['finish'] - range['start'] + 1 >= MAX_NUMBER_ITEMS:
        range['finish'] = range['start'] + MAX_NUMBER_ITEMS - 1
        filtered = items[range['start']:range['finish'] + 1]

    request.response.headers['Content-Range'] = 'items %d-%d/%d' % (range['start'], range['finish'], len(items))
    request.response.headers['X-Content-Range'] = request.response.headers['Content-Range']
    return filtered