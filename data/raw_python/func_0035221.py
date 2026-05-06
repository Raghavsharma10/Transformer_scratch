def hello_user(api_client):
    """Use an authorized client to fetch and print profile information.
    Parameters
        api_client (LyftRidesClient)
            An LyftRidesClient with OAuth 2.0 credentials.
    """

    try:
        response = api_client.get_user_profile()

    except (ClientError, ServerError) as error:
        fail_print(error)
        return

    else:
        profile = response.json
        user_id = profile.get('id')
        message = 'Hello. Successfully granted access token to User ID {}.'.format(user_id)
        success_print(message)