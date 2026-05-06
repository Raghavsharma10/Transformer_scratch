def make_heartbeat(port, path, peer_uid, node_uid, app_id):
    """
    Prepares the heart beat UDP packet

    Format : Little endian
    * Kind of beat (1 byte)
    * Herald HTTP server port (2 bytes)
    * Herald HTTP servlet path length (2 bytes)
    * Herald HTTP servlet path (variable, UTF-8)
    * Peer UID length (2 bytes)
    * Peer UID (variable, UTF-8)
    * Node UID length (2 bytes)
    * Node UID (variable, UTF-8)
    * Application ID length (2 bytes)
    * Application ID (variable, UTF-8)

    :param port: The port to access the Herald HTTP server
    :param path: The path to the Herald HTTP servlet
    :param peer_uid: The UID of the peer
    :param node_uid: The UID of the node
    :param app_id: Application ID
    :return: The heart beat packet content (byte array)
    """
    # Type and port...
    packet = struct.pack("<BBH", PACKET_FORMAT_VERSION, PACKET_TYPE_HEARTBEAT, port)
    for string in (path, peer_uid, node_uid, app_id):
        # Strings...
        string_bytes = to_bytes(string)
        packet += struct.pack("<H", len(string_bytes))
        packet += string_bytes

    return packet