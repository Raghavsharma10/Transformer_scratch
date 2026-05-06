def make_hash_id():
    """
    Compute the `datetime.now` based SHA-1 hash of a string.

    :return: Returns the sha1 hash as a string.
    :rtype: str
    """
    today = datetime.datetime.now().strftime(DATETIME_FORMAT)
    return hashlib.sha1(today.encode('utf-8')).hexdigest()