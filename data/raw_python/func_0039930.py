async def certifiers_of(client: Client, search: str) -> dict:
    """
    GET UID/Public key certifiers

    :param client: Client to connect to the api
    :param search: UID or public key
    :return:
    """
    return await client.get(MODULE + '/certifiers-of/%s' % search, schema=CERTIFICATIONS_SCHEMA)