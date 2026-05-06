def search_repository_configuration(url, page_size=10, page_index=0, sort=""):
    """
    Search for Repository Configurations based on internal or external url
    """
    content = search_repository_configuration_raw(url, page_size, page_index, sort)
    if content:
        return utils.format_json_list(content)