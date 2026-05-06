async def main():
    """
    Main code (synchronous requests)
    """
    # Create Client from endpoint string in Duniter format
    client = Client(ES_CORE_ENDPOINT)

    # Get the current node (direct REST GET request)
    print("\nGET g1-test/block/current/_source:")
    response = await client.get('g1-test/block/current/_source')
    print(response)

    # Get the node number 2 with only selected fields (direct REST GET request)
    print("\nGET g1-test/block/2/_source:")
    response = await client.get('g1-test/block/2/_source', {'_source': 'number,hash,dividend,membersCount'})
    print(response)

    # Close client aiohttp session
    await client.close()

    # Create Client from endpoint string in Duniter format
    client = Client(ES_USER_ENDPOINT)

    # prompt entry
    pubkey = input("\nEnter a public key to get the user profile: ")

    # Get the profil of a public key (direct REST GET request)
    print("\nGET user/profile/{0}/_source:".format(pubkey))
    response = await client.get('user/profile/{0}/_source'.format(pubkey.strip(' \n')))
    print(response)

    # Close client aiohttp session
    await client.close()