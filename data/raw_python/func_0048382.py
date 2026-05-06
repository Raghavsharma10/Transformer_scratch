def get_pseudo_salt(length, *args):
    """
    generate a pseudo salt (used, if user is wrong)
    """
    temp = "".join([arg for arg in args])
    return hash_hexdigest(temp)[:length]