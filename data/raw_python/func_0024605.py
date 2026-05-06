def from_payload(self, payload):
        """Init frame from binary data."""
        self.node_id = payload[0]
        self.state = payload[1]
        self.current_position = Parameter(payload[2:4])
        self.target = Parameter(payload[4:6])
        self.current_position_fp1 = Parameter(payload[6:8])
        self.current_position_fp2 = Parameter(payload[8:10])
        self.current_position_fp3 = Parameter(payload[10:12])
        self.current_position_fp4 = Parameter(payload[12:14])
        self.remaining_time = payload[14] * 256 + payload[15]
        # @VELUX: looks like your timestamp is wrong. Looks like
        # you are only transmitting the two lower bytes.
        self.timestamp = struct.unpack(">I", payload[16:20])[0]