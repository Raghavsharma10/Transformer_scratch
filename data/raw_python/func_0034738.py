def check_signature(signature, key, data):
    """Compute the HMAC signature and test against a given hash."""
    if isinstance(key, type(u'')):
        key = key.encode()

    digest = 'sha1=' + hmac.new(key, data, hashlib.sha1).hexdigest()

    # Covert everything to byte sequences
    if isinstance(digest, type(u'')):
        digest = digest.encode()
    if isinstance(signature, type(u'')):
        signature = signature.encode()

    return werkzeug.security.safe_str_cmp(digest, signature)