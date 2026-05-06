async def memberships(client: Client, search: str) -> dict:
    """
    GET list of Membership documents for UID/Public key

    :param client: Client to connect to the api
    :param search: UID/Public key
    :return:
    """
    return await client.get(MODULE + '/memberships/%s' % search, schema=MEMBERSHIPS_SCHEMA)