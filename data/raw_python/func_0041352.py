def make_lastbeat(peer_uid, app_id):
    """
    Prepares the last beat UDP packet (when the peer is going away)

    Format : Little endian
    * Kind of beat (1 byte)
    * Peer UID length (2 bytes)
    * Peer UID (variable, UTF-8)
    * Application ID length (2 bytes)
    * Application ID (variable, UTF-8)

    :param peer_uid: Peer UID
    :param app_id: Application ID
    :return: The last beat packet content (byte array)
    """
    packet = struct.pack("<BB", PACKET_FORMAT_VERSION, PACKET_TYPE_LASTBEAT)
    for string in (peer_uid, app_id):
        string_bytes = to_bytes(string)
        packet += struct.pack("<H", len(string_bytes))
        packet += string_bytes

    return packet