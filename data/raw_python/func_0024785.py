def from_payload(self, payload):
        """Init frame from binary data."""
        self.node_id = payload[0]
        self.order = payload[1] * 256 + payload[2]
        self.placement = payload[3]
        self.name = bytes_to_string(payload[4:68])
        self.velocity = Velocity(payload[68])
        self.node_type = NodeTypeWithSubtype(payload[69] * 256 + payload[70])
        self.product_group = payload[71]
        self.product_type = payload[72]
        self.node_variation = NodeVariation(payload[73])
        self.power_mode = payload[74]
        self.build_number = payload[75]
        self._serial_number = payload[76:84]
        self.state = payload[84]
        self.current_position = Parameter(payload[85:87])
        self.target = Parameter(payload[87:89])
        self.current_position_fp1 = Parameter(payload[89:91])
        self.current_position_fp2 = Parameter(payload[91:93])
        self.current_position_fp3 = Parameter(payload[93:95])
        self.current_position_fp4 = Parameter(payload[95:97])
        self.remaining_time = payload[97] * 256 + payload[98]
        self.timestamp = struct.unpack(">I", payload[99:103])[0]
        self.alias_array = AliasArray(payload[103:125])