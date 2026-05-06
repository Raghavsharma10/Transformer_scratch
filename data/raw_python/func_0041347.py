async def blocks(client: Client, count: int, start: int) -> list:
    """
    GET list of blocks from the blockchain

    :param client: Client to connect to the api
    :param count: Number of blocks
    :param start: First block number
    :return:
    """
    assert type(count) is int
    assert type(start) is int

    return await client.get(MODULE + '/blocks/%d/%d' % (count, start), schema=BLOCKS_SCHEMA)