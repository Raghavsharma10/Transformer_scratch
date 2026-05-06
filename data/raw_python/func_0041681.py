async def process(client: Client, transaction_signed_raw: str) -> ClientResponse:
    """
    POST a transaction raw document

    :param client: Client to connect to the api
    :param transaction_signed_raw: Transaction signed raw document
    :return:
    """
    return await client.post(MODULE + '/process', {'transaction': transaction_signed_raw}, rtype=RESPONSE_AIOHTTP)