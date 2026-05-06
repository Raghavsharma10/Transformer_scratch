def rand_str(charset, length=32):
    """
    Generate random string.
    """
    return "".join([random.choice(charset) for _ in range(length)])