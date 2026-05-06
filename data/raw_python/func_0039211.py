def remove_nullchars(block):
    """Strips NULL chars taking care of bytes alignment."""
    data = block.lstrip(b'\00')

    padding = b'\00' * ((len(block) - len(data)) % 8)

    return padding + data