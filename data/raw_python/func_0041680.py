async def history(client: Client, pubkey: str) -> dict:
    """
    Get transactions history of public key

    :param client: Client to connect to the api
    :param pubkey: Public key
    :return:
    """
    return await client.get(MODULE + '/history/%s' % pubkey, schema=HISTORY_SCHEMA)