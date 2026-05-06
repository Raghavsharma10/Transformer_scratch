def _from_hex_digest(digest):
    """Convert hex digest to sequence of bytes."""
    return "".join(
        [chr(int(digest[x : x + 2], 16)) for x in range(0, len(digest), 2)]
    )