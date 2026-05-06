def example_sync_client(api_client):
    """Example sync client use with.
    """

    try:
        pprint(api_client.echo())
    except errors.RequestError as exc:
        log.exception('Exception occurred: %s', exc)