def check_password(password: str, encrypted: str) -> bool:
    """ Check a plaintext password against a hashed password. """
    # some old passwords have {crypt} in lower case, and passlib wants it to be
    # in upper case.
    if encrypted.startswith("{crypt}"):
        encrypted = "{CRYPT}" + encrypted[7:]
    return pwd_context.verify(password, encrypted)