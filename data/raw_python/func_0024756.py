def decode(raw):
    """Decode SLIP message."""
    return raw \
        .replace(bytes([SLIP_ESC, SLIP_ESC_END]), bytes([SLIP_END])) \
        .replace(bytes([SLIP_ESC, SLIP_ESC_ESC]), bytes([SLIP_ESC]))