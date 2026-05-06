def encode(raw):
    """Encode SLIP message."""
    return raw \
        .replace(bytes([SLIP_ESC]), bytes([SLIP_ESC, SLIP_ESC_ESC])) \
        .replace(bytes([SLIP_END]), bytes([SLIP_ESC, SLIP_ESC_END]))