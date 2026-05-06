def delete_api_key(awsclient, api_key):
    """Remove API key.

    :param api_key:
    """
    _sleep()
    client_api = awsclient.get_client('apigateway')
    print('delete api key: %s' % api_key)

    response = client_api.delete_api_key(
        apiKey=api_key
    )

    print(json2table(response))