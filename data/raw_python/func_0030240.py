def list_running_builds(page_size=200, page_index=0, sort=""):
    """
    List all RunningBuilds
    """
    content = list_running_builds_raw(page_size, page_index, sort)
    if content:
        return utils.format_json_list(content)