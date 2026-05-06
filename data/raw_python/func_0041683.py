async def blocks(client: Client, pubkey: str, start: int, end: int) -> dict:
    """
    GET public key transactions history between start and end block number

    :param client: Client to connect to the api
    :param pubkey: Public key
    :param start: Start from block number
    :param end: End to block number
    :return:
    """
    return await client.get(MODULE + '/history/%s/blocks/%s/%s' % (pubkey, start, end), schema=HISTORY_SCHEMA)