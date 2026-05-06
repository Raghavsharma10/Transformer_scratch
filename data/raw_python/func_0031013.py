def example_async_client(api_client):
    """Example async client.
    """

    try:
        pprint((yield from api_client.echo()))
    except errors.RequestError as exc:
        log.exception('Exception occurred: %s', exc)

    yield gen.Task(lambda *args, **kwargs: ioloop.IOLoop.current().stop())