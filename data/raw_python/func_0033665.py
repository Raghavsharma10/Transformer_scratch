def parse_payload(data, payload_fmt, *payload_names):
    """
    Parses a bytestring of Lifx payload data (the bytes after the common
    fields), as returned by `parse_packet`. Returns a dictionary where the keys
    are from `payload_names` and the values are the corresponding values from
    the bytestring.
    """
    payload = struct.unpack(payload_fmt, data)
    return dict(zip(payload_names, payload))