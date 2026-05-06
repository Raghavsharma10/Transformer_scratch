def get_build_configuration_set_raw(id=None, name=None):
    """
    Get a specific BuildConfigurationSet by name or ID
    """
    found_id = common.set_id(pnc_api.build_group_configs, id, name)
    response = utils.checked_api_call(pnc_api.build_group_configs, 'get_specific', id=found_id)
    if response:
        return response.content