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

    # create keys from credentials
    key = SigningKey.from_credentials(salt, password)
    pubkey_from = key.pubkey

    # prompt entry
    pubkey_to = input("Enter recipient pubkey: ")

    # capture current block to get version and currency and blockstamp
    current_block = await client(bma.blockchain.current)

    # capture sources of account
    response = await client(bma.tx.sources, pubkey_from)

    if len(response['sources']) == 0:
        print("no sources found for account %s" % pubkey_to)
        exit(1)

    # get the first source
    source = response['sources'][0]

    # create the transaction document
    transaction = get_transaction_document(current_block, source, pubkey_from, pubkey_to)

    # sign document
    transaction.sign([key])

    # send the Transaction document to the node
    response = await client(bma.tx.process, transaction.signed_raw())

    if response.status == 200:
        print(await response.text())
    else:
        print("Error while publishing transaction: {0}".format(await response.text()))

    # Close client aiohttp session
    await client.close()