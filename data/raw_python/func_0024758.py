def get_next_slip(raw):
    """
    Get the next slip packet from raw data.

    Returns the extracted packet plus the raw data with the remaining data stream.
    """
    if not is_slip(raw):
        return None, raw
    length = raw[1:].index(SLIP_END)
    slip_packet = decode(raw[1:length+1])
    new_raw = raw[length+2:]
    return slip_packet, new_raw