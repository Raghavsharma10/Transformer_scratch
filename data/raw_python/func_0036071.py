def raw_data(self, value):
        """Set the base64 encoded data using a raw value or file object."""
        if value:
            try:
                value = value.read()
            except AttributeError:
                pass
            b64 = base64.b64encode(value.encode('utf-8'))
            self.data = b64.decode('utf-8')