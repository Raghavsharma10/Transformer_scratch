async def add(client: Client, identity_signed_raw: str) -> ClientResponse:
    """
    POST identity raw document

    :param client: Client to connect to the api
    :param identity_signed_raw: Identity raw document
    :return:
    """
    return await client.post(MODULE + '/add', {'identity': identity_signed_raw}, rtype=RESPONSE_AIOHTTP)