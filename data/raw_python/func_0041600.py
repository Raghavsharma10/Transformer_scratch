async def peer(client: Client, peer_signed_raw: str) -> ClientResponse:
    """
    POST a Peer signed raw document

    :param client: Client to connect to the api
    :param peer_signed_raw: Peer signed raw document
    :return:
    """
    return await client.post(MODULE + '/peering/peers', {'peer': peer_signed_raw}, rtype=RESPONSE_AIOHTTP)