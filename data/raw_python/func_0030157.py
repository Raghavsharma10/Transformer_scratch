def create_build_configuration_set_raw(**kwargs):
    """
    Create a new BuildConfigurationSet.
    """
    config_set = _create_build_config_set_object(**kwargs)
    response = utils.checked_api_call(pnc_api.build_group_configs, 'create_new', body=config_set)
    if response:
        return response.content