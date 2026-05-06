def password_hash(password, password_salt=None):
    """Hashes a specified password"""
    password_salt = password_salt or oz.settings["session_salt"]
    salted_password = password_salt + password
    return "sha256!%s" % hashlib.sha256(salted_password.encode("utf-8")).hexdigest()