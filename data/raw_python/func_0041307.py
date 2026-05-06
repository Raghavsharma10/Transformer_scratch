async def main():
    """
    Main code
    """
    # Create Client from endpoint string in Duniter format
    client = Client(BMAS_ENDPOINT)

    # Get the node summary infos to test the connection
    response = await client(bma.node.summary)
    print(response)

    # prompt hidden user entry
    salt = getpass.getpass("Enter your passphrase (salt): ")

    # prompt hidden user entry
    password = getpass.getpass("Enter your password: ")

    # create key from credentials
    key = SigningKey.from_credentials(salt, password)
    pubkey_from = key.pubkey

    # prompt entry
    pubkey_to = input("Enter certified pubkey: ")

    # capture current block to get version and currency and blockstamp
    current_block = await client(bma.blockchain.current)

    # create our Identity document to sign the Certification document
    identity = await get_identity_document(client, current_block, pubkey_to)

    # send the Certification document to the node
    certification = get_certification_document(current_block, identity, pubkey_from)

    # sign document
    certification.sign([key])

    # Here we request for the path wot/certify
    response = await client(bma.wot.certify, certification.signed_raw())

    if response.status == 200:
        print(await response.text())
    else:
        print("Error while publishing certification: {0}".format(await response.text()))

    # Close client aiohttp session
    await client.close()