async def block(client: Client, number: int = 0, block_raw: str = None, signature: str = None) -> Union[dict,
                                                                                                        ClientResponse]:
    """
    GET/POST a block from/to the blockchain

    :param client: Client to connect to the api
    :param number: Block number to get
    :param block_raw: Block document to post
    :param signature: Signature of the block document issuer
    :return:
    """
    # POST block
    if block_raw is not None and signature is not None:
        return await client.post(MODULE + '/block', {'block': block_raw, 'signature': signature},
                                 rtype=RESPONSE_AIOHTTP)
    # GET block
    return await client.get(MODULE + '/block/%d' % number, schema=BLOCK_SCHEMA)