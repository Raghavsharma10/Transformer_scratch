async def hardship(client: Client, pubkey: str) -> dict:
    """
    GET hardship level for given member's public key for writing next block

    :param client: Client to connect to the api
    :param pubkey:  Public key of the member
    :return:
    """
    return await client.get(MODULE + '/hardship/%s' % pubkey, schema=HARDSHIP_SCHEMA)