def parse_packet(data, format=None):
    """
    Parses a Lifx data packet (as a bytestring), returning into a Header object
    for the fields that are common to all data packets, and a bytestring of
    payload data for the type-specific fields (suitable for passing to
    `parse_payload`).
    """
    unpacked = struct.unpack(BASE_FORMAT, data[:_FORMAT_SIZE])
    psize, protocol, mac, gateway, time, ptype = unpacked
    header = Header(psize, protocol, mac, gateway, time, ptype)
    return header, data[_FORMAT_SIZE:]