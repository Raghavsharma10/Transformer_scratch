def get_membership_document(membership_type: str, current_block: dict, identity: Identity, salt: str,
                            password: str) -> Membership:
    """
    Get a Membership document

    :param membership_type: "IN" to ask for membership or "OUT" to cancel membership
    :param current_block: Current block data
    :param identity: Identity document
    :param salt: Passphrase of the account
    :param password: Password of the account

    :rtype: Membership
    """

    # get current block BlockStamp
    timestamp = BlockUID(current_block['number'], current_block['hash'])

    # create keys from credentials
    key = SigningKey.from_credentials(salt, password)

    # create identity document
    membership = Membership(
        version=10,
        currency=current_block['currency'],
        issuer=key.pubkey,
        membership_ts=timestamp,
        membership_type=membership_type,
        uid=identity.uid,
        identity_ts=identity.timestamp,
        signature=None
    )

    # sign document
    membership.sign([key])

    return membership