def list_builds(page_size=200, page_index=0, sort="", q=""):
    """
    List all builds
    :param page_size: number of builds returned per query
    :param sort: RSQL sorting query
    :param q: RSQL query
    :return:
    """
    response = utils.checked_api_call(pnc_api.builds_running, 'get_all', page_size=page_size, page_index=page_index, sort=sort, q=q)
    if response:
        return response.content