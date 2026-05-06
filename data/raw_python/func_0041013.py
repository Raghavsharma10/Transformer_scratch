async def main():
    """
    Main code
    """
    # Create Client from endpoint string in Duniter format
    client = Client(BMAS_ENDPOINT)

    # Get the node summary infos to test the connection
    response = await client(bma.node.summary)
    print(response)

    # capture current block to get version and currency and blockstamp
    current_block = await client(bma.blockchain.current)

    # prompt entry
    uid = input("Enter your Unique IDentifier (pseudonym): ")

    # prompt hidden user entry
    salt = getpass.getpass("Enter your passphrase (salt): ")

    # prompt hidden user entry
    password = getpass.getpass("Enter your password: ")

    # create our signed identity document
    identity = get_identity_document(current_block, uid, salt, password)

    # send the identity document to the node
    response = await client(bma.wot.add, identity.signed_raw())
    if response.status == 200:
        print(await response.text())
    else:
        print("Error while publishing identity : {0}".format(await response.text()))

    # Close client aiohttp session
    await client.close()