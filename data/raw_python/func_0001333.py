def decrypt(data, digest=True):
    """Decrypt provided data."""
    alg, _, data = data.rpartition("$")
    if not alg:
        return data
    data = _from_hex_digest(data) if digest else data
    try:
        return implementations["decryption"][alg](
            data, implementations["get_key"]()
        )
    except KeyError:
        raise CryptError("Can not decrypt key for algorithm: %s" % alg)