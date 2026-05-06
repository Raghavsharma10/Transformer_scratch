async def sources(client: Client, pubkey: str) -> dict:
    """
    GET transaction sources

    :param client: Client to connect to the api
    :param pubkey: Public key
    :return:
    """
    return await client.get(MODULE + '/sources/%s' % pubkey, schema=SOURCES_SCHEMA)