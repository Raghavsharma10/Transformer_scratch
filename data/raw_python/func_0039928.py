async def revoke(client: Client, revocation_signed_raw: str) -> ClientResponse:
    """
    POST revocation document

    :param client: Client to connect to the api
    :param revocation_signed_raw: Certification raw document
    :return:
    """
    return await client.post(MODULE + '/revoke', {'revocation': revocation_signed_raw}, rtype=RESPONSE_AIOHTTP)