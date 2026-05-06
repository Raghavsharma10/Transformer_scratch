def search_repository_configuration_raw(url, page_size=10, page_index=0, sort=""):
    """
    Search for Repository Configurations based on internal or external url
    """
    response = utils.checked_api_call(pnc_api.repositories, 'search', page_size=page_size, page_index=page_index, sort=sort, search=url)
    if response:
        return response.content