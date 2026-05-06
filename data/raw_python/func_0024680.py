def calc_crc(raw):
    """Calculate cyclic redundancy check (CRC)."""
    crc = 0
    for sym in raw:
        crc = crc ^ int(sym)
    return crc