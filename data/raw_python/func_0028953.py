def create_api_key(awsclient, api_name, api_key_name):
    """Create a new API key as reference for api.conf.

    :param api_name:
    :param api_key_name:
    :return: api_key
    """
    _sleep()
    client_api = awsclient.get_client('apigateway')
    print('create api key: %s' % api_key_name)

    response = client_api.create_api_key(
        name=api_key_name,
        description='Created for ' + api_name,
        enabled=True
    )

    #print(json2table(response))

    print('Add this api key \'%s\' to your api.conf' % response['id'])
    return response['id']