def all_pages(method, request, accessor, cond=None):
    """Helper to process all pages using botocore service methods (exhausts NextToken).
    note: `cond` is optional... you can use it to make filtering more explicit
    if you like. Alternatively you can do the filtering in the `accessor` which
    is perfectly fine, too
    Note: lambda uses a slightly different mechanism so there is a specific version in
    ramuda_utils.

    :param method: service method
    :param request: request dictionary for service call
    :param accessor: function to extract data from each response
    :param cond: filter function to return True / False based on a response
    :return: list of collected resources
    """
    if cond is None:
        cond = lambda x: True
    result = []
    next_token = None
    while True:
        if next_token:
            request['nextToken'] = next_token
        response = method(**request)
        if cond(response):
            data = accessor(response)
            if data:
                if isinstance(data, list):
                    result.extend(data)
                else:
                    result.append(data)
        if 'nextToken' not in response:
            break
        next_token = response['nextToken']

    return result