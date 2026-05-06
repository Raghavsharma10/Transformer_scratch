def _update_stage(awsclient, api_id, stage_name, method_settings):
    """Helper to apply method_settings to stage

    :param awsclient:
    :param api_id:
    :param stage_name:
    :param method_settings:
    :return:
    """
    # settings docs in response: https://botocore.readthedocs.io/en/latest/reference/services/apigateway.html#APIGateway.Client.update_stage
    client_api = awsclient.get_client('apigateway')
    operations = _convert_method_settings_into_operations(method_settings)
    if operations:
        print('update method settings for stage')
        _sleep()
        response = client_api.update_stage(
            restApiId=api_id,
            stageName=stage_name,
            patchOperations=operations)