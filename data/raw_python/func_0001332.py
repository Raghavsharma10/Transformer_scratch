def encrypt(data, digest=True):
    """Perform encryption of provided data."""
    alg = get_best_algorithm()
    enc = implementations["encryption"][alg](
        data, implementations["get_key"]()
    )
    return "%s$%s" % (alg, (_to_hex_digest(enc) if digest else enc))