def get_signed_raw_revocation_document(identity: Identity, salt: str, password: str) -> str:
    """
    Generate account revocation document for given identity

    :param identity: Self Certification of the identity
    :param salt: Salt
    :param password: Password

    :rtype: str
    """
    revocation = Revocation(PROTOCOL_VERSION, identity.currency, identity, "")

    key = SigningKey.from_credentials(salt, password)
    revocation.sign([key])
    return revocation.signed_raw()