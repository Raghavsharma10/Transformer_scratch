def get_revision_of_build_configuration(revision_id, id=None, name=None):
    """
    Get a specific audited revision of a BuildConfiguration
    """
    data = get_revision_of_build_configuration_raw(revision_id, id, name)
    if data:
        return utils.format_json_list(data)