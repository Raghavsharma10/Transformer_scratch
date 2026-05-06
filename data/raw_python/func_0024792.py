def get_payload(self):
        """Return Payload."""
        ret = self._software_version
        ret += bytes([self.hardware_version, self.product_group, self.product_type])
        return ret