def get_random_hex(length):
    """
    Return random hex string of a given length
    """
    if length <= 0:
        return ''
    return hexify(random.randint(pow(2, length*2), pow(2, length*4)))[0:length]