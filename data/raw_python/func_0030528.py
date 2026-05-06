def list_revisions_of_build_configuration(id=None, name=None, page_size=200, page_index=0, sort=""):
    """
    List audited revisions of a BuildConfiguration
    """
    data = list_revisions_of_build_configuration_raw(id, name, page_size, page_index, sort)
    if data:
        return utils.format_json_list(data)