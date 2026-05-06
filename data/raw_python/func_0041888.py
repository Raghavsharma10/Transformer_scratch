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

    # prompt public key
    pubkey = input("Enter your public key: ")

    # init signer instance
    signer = SigningKey.from_credentials(salt, password)

    # check public key
    if signer.pubkey != pubkey:
        print("Bad credentials!")
        exit(0)

    # capture current block to get currency name
    current_block = await client(bma.blockchain.current)

    # create our Identity document to sign the revoke document
    identity_document = await get_identity_document(client, current_block['currency'], pubkey)

    # get the revoke document
    revocation_signed_raw_document = get_signed_raw_revocation_document(identity_document, salt, password)

    # save revoke document in a file
    fp = open(REVOCATION_DOCUMENT_FILE_PATH, 'w')
    fp.write(revocation_signed_raw_document)
    fp.close()

    # document saved
    print("Revocation document saved in %s" % REVOCATION_DOCUMENT_FILE_PATH)

    # Close client aiohttp session
    await client.close()