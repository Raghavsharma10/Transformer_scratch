def rand_elem(seq, n=None):
    """returns a random element from seq n times. If n is None, it continues indefinitly"""
    return map(random.choice, repeat(seq, n) if n is not None else repeat(seq))