def std_hash(word, salt):
    """Generates a cryptographically strong (sha512) hash with this nodes
    salt added."""

    try:
        password = word.encode('utf-8')
    except UnicodeDecodeError:
        password = word

    word_hash = sha512(password)
    word_hash.update(salt)
    hex_hash = word_hash.hexdigest()

    return hex_hash