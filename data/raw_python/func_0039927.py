async def certify(client: Client, certification_signed_raw: str) -> ClientResponse:
    """
    POST certification raw document

    :param client: Client to connect to the api
    :param certification_signed_raw: Certification raw document
    :return:
    """
    return await client.post(MODULE + '/certify', {'cert': certification_signed_raw}, rtype=RESPONSE_AIOHTTP)