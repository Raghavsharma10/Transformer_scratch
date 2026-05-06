def check_password_hash(password, password_hash, salt, N=1 << 14, r=8, p=1, buflen=64):
    """
    Given a password, hash, salt this function verifies the password is equal to hash/salt.

    Args:
       - ``password``: The password to perform check on.

    Returns:
       - ``bool``
    """
    candidate_hash = generate_password_hash(password, salt, N, r, p, buflen)

    return safe_str_cmp(password_hash, candidate_hash)