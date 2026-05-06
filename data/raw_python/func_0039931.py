async def requirements(client: Client, search: str) -> dict:
    """
    GET list of requirements for a given UID/Public key

    :param client: Client to connect to the api
    :param search: UID or public key
    :return:
    """
    return await client.get(MODULE + '/requirements/%s' % search, schema=REQUIREMENTS_SCHEMA)