async def membership(client: Client, membership_signed_raw: str) -> ClientResponse:
    """
    POST a Membership document

    :param client: Client to connect to the api
    :param membership_signed_raw: Membership signed raw document
    :return:
    """
    return await client.post(MODULE + '/membership', {'membership': membership_signed_raw}, rtype=RESPONSE_AIOHTTP)