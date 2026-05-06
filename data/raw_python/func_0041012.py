def get_identity_document(current_block: dict, uid: str, salt: str, password: str) -> Identity:
    """
    Get an Identity document

    :param current_block: Current block data
    :param uid: Unique IDentifier
    :param salt: Passphrase of the account
    :param password: Password of the account

    :rtype: Identity
    """

    # get current block BlockStamp
    timestamp = BlockUID(current_block['number'], current_block['hash'])

    # create keys from credentials
    key = SigningKey.from_credentials(salt, password)

    # create identity document
    identity = Identity(
        version=10,
        currency=current_block['currency'],
        pubkey=key.pubkey,
        uid=uid,
        ts=timestamp,
        signature=None
    )

    # sign document
    identity.sign([key])

    return identity