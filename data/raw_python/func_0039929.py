async def lookup(client: Client, search: str) -> dict:
    """
    GET UID/Public key data

    :param client: Client to connect to the api
    :param search: UID or public key
    :return:
    """
    return await client.get(MODULE + '/lookup/%s' % search, schema=LOOKUP_SCHEMA)