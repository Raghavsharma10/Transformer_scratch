def random_string(length=6):
    """Create a random 6 character string.

    note: in case you use this function in a test during test together with
    an awsclient then this function is altered so you get reproducible results
    that will work with your recorded placebo json files (see helpers_aws.py).
    """
    return ''.join([random.choice(string.ascii_lowercase) for i in range(length)])