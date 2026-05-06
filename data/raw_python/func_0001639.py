def _unique_names():
    """Generates unique sequences of bytes.
    """
    characters = ("abcdefghijklmnopqrstuvwxyz"
                  "0123456789")
    characters = [characters[i:i + 1] for i in irange(len(characters))]
    rng = random.Random()
    while True:
        letters = [rng.choice(characters) for i in irange(10)]
        yield ''.join(letters)