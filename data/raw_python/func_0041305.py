async def get_identity_document(client: Client, current_block: dict, pubkey: str) -> Identity:
    """
    Get the identity document of the pubkey

    :param client: Client to connect to the api
    :param current_block: Current block data
    :param pubkey: UID/Public key

    :rtype: Identity
    """
    # Here we request for the path wot/lookup/pubkey
    lookup_data = await client(bma.wot.lookup, pubkey)

    # init vars
    uid = None
    timestamp = BlockUID.empty()
    signature = None

    # parse results
    for result in lookup_data['results']:
        if result["pubkey"] == pubkey:
            uids = result['uids']
            for uid_data in uids:
                # capture data
                timestamp = BlockUID.from_str(uid_data["meta"]["timestamp"])
                uid = uid_data["uid"]
                signature = uid_data["self"]

            # return self-certification document
            return Identity(
                version=10,
                currency=current_block['currency'],
                pubkey=pubkey,
                uid=uid,
                ts=timestamp,
                signature=signature
            )