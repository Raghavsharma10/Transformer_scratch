def export_to_swagger(awsclient, api_name, stage_name, api_description,
                      lambdas, custom_hostname=False, custom_base_path=False):
    """Export the API design as swagger file. 
    
    :param api_name: 
    :param stage_name: 
    :param api_description: 
    :param lambdas: 
    :param custom_hostname: 
    :param custom_base_path: 
    """
    print('Exporting to swagger...')

    api = _api_by_name(awsclient, api_name)
    if api is not None:

        print(json2table(api))
        api_id = api['id']
        client_api = awsclient.get_client('apigateway')
        template_variables = _template_variables_to_dict(
            client_api,
            api_name,
            api_description,
            stage_name,
            api_id,
            lambdas,
            custom_hostname,
            custom_base_path)
        content = _compile_template(SWAGGER_FILE, template_variables)
        swagger_file = open('swagger_export.yaml', 'w')

        swagger_file.write(content)
    else:
        print('API name unknown')