def from_payload(self, payload):
        """Init frame from binary data."""
        self._software_version = payload[0:6]
        self.hardware_version = payload[6]
        self.product_group = payload[7]
        self.product_type = payload[8]