def random_string(length):
    """
    Generates a random alphanumeric string
    """
    # avoid things that could be mistaken ex: 'I' and '1'
    letters = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join([random.choice(letters) for _ in range(length)])