def deploy_api(awsclient, api_name, api_description, stage_name, api_key,
               lambdas, cache_cluster_enabled, cache_cluster_size, method_settings=None):
    """Deploy API Gateway to AWS cloud.

    :param awsclient:
    :param api_name:
    :param api_description:
    :param stage_name:
    :param api_key:
    :param lambdas:
    :param cache_cluster_enabled:
    :param cache_cluster_size:
    :param method_settings:
    """
    if not _api_exists(awsclient, api_name):
        if os.path.isfile(SWAGGER_FILE):
            # this does an import from swagger file
            # the next step does not make sense since there is a check in
            # _import_from_swagger for if api is existent!
            # _create_api(api_name=api_name, api_description=api_description)
            _import_from_swagger(awsclient, api_name, api_description,
                                 stage_name, lambdas)
        else:
            print('No swagger file (%s) found' % SWAGGER_FILE)

        api = _api_by_name(awsclient, api_name)
        if api is not None:
            _ensure_lambdas_permissions(awsclient, lambdas, api)
            _create_deployment(awsclient, api_name, stage_name, cache_cluster_enabled, cache_cluster_size)
            _update_stage(awsclient, api['id'], stage_name, method_settings)
            _wire_api_key(awsclient, api_name, api_key, stage_name)
        else:
            print('API name unknown')
    else:
        if os.path.isfile(SWAGGER_FILE):
            _update_from_swagger(awsclient, api_name, api_description,
                                 stage_name, lambdas)
        else:
            _update_api()

        api = _api_by_name(awsclient, api_name)
        if api is not None:
            _ensure_lambdas_permissions(awsclient, lambdas, api)
            _create_deployment(awsclient, api_name, stage_name, cache_cluster_enabled, cache_cluster_size)
            _update_stage(awsclient, api['id'], stage_name, method_settings)
        else:
            print('API name unknown')