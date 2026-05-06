def list_licenses(page_size=200, page_index=0, sort="", q=""):
    """
    List all Licenses
    """
    response = utils.checked_api_call(pnc_api.licenses, 'get_all', page_size=page_size, page_index=page_index, sort=sort, q=q)
    if response:
        return utils.format_json_list(response.content)