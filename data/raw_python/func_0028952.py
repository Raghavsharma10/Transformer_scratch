def delete_api(awsclient, api_name):
    """Delete the API.

    :param api_name:
    """
    _sleep()
    client_api = awsclient.get_client('apigateway')

    print('deleting api: %s' % api_name)
    api = _api_by_name(awsclient, api_name)

    if api is not None:
        print(json2table(api))

        response = client_api.delete_rest_api(
            restApiId=api['id']
        )

        print(json2table(response))
    else:
        print('API name unknown')