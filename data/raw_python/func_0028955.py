def list_api_keys(awsclient):
    """Print the defined API keys.
    """
    _sleep()
    client_api = awsclient.get_client('apigateway')
    print('listing api keys')

    response = client_api.get_api_keys()['items']

    for item in response:
        print(json2table(item))