def list_projects(page_size=200, page_index=0, sort="", q=""):
    """
    List all Projects
    """
    content = list_projects_raw(page_size=page_size, page_index=page_index, sort=sort, q=q)
    if content:
        return utils.format_json_list(content)