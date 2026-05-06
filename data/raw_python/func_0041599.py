async def peers(client: Client, leaves: bool = False, leaf: str = "") -> dict:
    """
    GET peering entries of every node inside the currency network

    :param client: Client to connect to the api
    :param leaves: True if leaves should be requested
    :param leaf: True if leaf should be requested
    :return:
    """
    if leaves is True:
        return await client.get(MODULE + '/peering/peers', {"leaves": "true"}, schema=PEERS_SCHEMA)
    else:
        return await client.get(MODULE + '/peering/peers', {"leaf": leaf}, schema=PEERS_SCHEMA)