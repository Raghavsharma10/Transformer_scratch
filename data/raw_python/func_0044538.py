def generate_password_hash(password, salt, N=1 << 14, r=8, p=1, buflen=64):
    """
    Generate password hash givin the password string and salt.

    Args:
        - ``password``: Password string.
        - ``salt`` : Random base64 encoded string.

    Optional args:
        - ``N`` : the CPU cost, must be a power of 2 greater than 1, defaults to 1 << 14.
        - ``r`` : the memory cost, defaults to 8.
        - ``p`` : the parallelization parameter, defaults to 1.

    The parameters r, p, and buflen must satisfy r * p < 2^30 and
    buflen <= (2^32 - 1) * 32.

    The recommended parameters for interactive logins as of 2009 are N=16384,
    r=8, p=1. Remember to use a good random salt.

    Returns:
        - base64 encoded scrypt hash.
    """
    if PYTHON2:
        password = password.encode('utf-8')
        salt = salt.encode('utf-8')
    pw_hash = scrypt_hash(password, salt, N, r, p, buflen)
    return enbase64(pw_hash)