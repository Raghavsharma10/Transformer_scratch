async def identity_of(client: Client, search: str) -> dict:
    """
    GET Identity data written in the blockchain

    :param client: Client to connect to the api
    :param search: UID or public key
    :return:
    """
    return await client.get(MODULE + '/identity-of/%s' % search, schema=IDENTITY_OF_SCHEMA)