def pack(self):
        """
        Pack the frame into a string according to the following scheme:

        +-+-+-+-+-------+-+-------------+-------------------------------+
        |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
        |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
        |N|V|V|V|       |S|             |   (if payload len==126/127)   |
        | |1|2|3|       |K|             |                               |
        +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
        |     Extended payload length continued, if payload len == 127  |
        + - - - - - - - - - - - - - - - +-------------------------------+
        |                               |Masking-key, if MASK set to 1  |
        +-------------------------------+-------------------------------+
        | Masking-key (continued)       |          Payload Data         |
        +-------------------------------- - - - - - - - - - - - - - - - +
        :                     Payload Data continued ...                :
        + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
        |                     Payload Data continued ...                |
        +---------------------------------------------------------------+
        """
        header = struct.pack('!B', (self.final << 7) | (self.rsv1 << 6)
                                   | (self.rsv2 << 5) | (self.rsv3 << 4)
                                   | (self.opcode & 0xf))
        mask = bool(self.masking_key) << 7
        payload_len = len(self.payload)

        if payload_len <= 125:
            header += struct.pack('!B', mask | payload_len)
        elif payload_len < (1 << 16):
            header += struct.pack('!BH', mask | 126, payload_len)
        elif payload_len < (1 << 63):
            header += struct.pack('!BQ', mask | 127, payload_len)
        else:
            # FIXME: RFC 6455 defines an action for this...
            raise Exception('the payload length is too damn high!')

        if mask:
            return header + self.masking_key + self.mask_payload()

        return header + self.payload