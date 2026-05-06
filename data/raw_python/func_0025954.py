def unpack(cls, rawpacket):
        """Instantiate `Packet` from binary string.

           :param rawpacket: TSIP pkt in binary format.
           :type rawpacket: String.

           `rawpacket` must already have framing (DLE...DLE/ETX) removed and
           byte stuffing reversed.

        """

        structs_ = get_structs_for_rawpacket(rawpacket)

        for struct_ in structs_:
            try:
                return cls(*struct_.unpack(rawpacket))
            except struct.error:
                raise
                # Try next one.
                pass

        # Packet ID 0xff is a pseudo-packet representing
        # packets unknown to `python-TSIP` in their raw format.
        #
        return cls(0xff, rawpacket)