def make_passwordmanager(schemes=None):
    """
    schemes contains a list of replace this list with the hash(es) you wish
    to support.
    this example sets pbkdf2_sha256 as the default,
    with support for legacy bcrypt hashes.

    :param schemes:
    :return: CryptContext()
    """
    from passlib.context import CryptContext

    if not schemes:
        schemes = ["pbkdf2_sha256", "bcrypt"]
    pwd_context = CryptContext(schemes=schemes, deprecated="auto")
    return pwd_context