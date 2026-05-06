def md5_for_file(f, block_size=2 ** 20):
    """Generate an MD5 has for a possibly large file by breaking it into
    chunks."""

    md5 = hashlib.md5()
    try:
        # Guess that f is a FLO.
        f.seek(0)

        return md5_for_stream(f, block_size=block_size)

    except AttributeError:
        # Nope, not a FLO. Maybe string?

        file_name = f
        with open(file_name, 'rb') as f:
            return md5_for_file(f, block_size)