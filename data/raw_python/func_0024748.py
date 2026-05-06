def build_frame(command, payload):
        """Build raw bytes from command and payload."""
        packet_length = 2 + len(payload) + 1
        ret = struct.pack("BB", 0, packet_length)
        ret += struct.pack(">H", command.value)
        ret += payload
        ret += struct.pack("B", calc_crc(ret))
        return ret