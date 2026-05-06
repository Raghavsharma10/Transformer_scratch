def to_dict(self):
        """
        Return a dict that can be serialised to JSON and sent to UpCloud's API.

        Uses the convenience attribute `os` for determining `action` and `storage`
        fields.
        """
        body = {
            'tier': self.tier,
            'title': self.title,
            'size': self.size,
        }

        # optionals

        if hasattr(self, 'address') and self.address:
            body['address'] = self.address

        if hasattr(self, 'zone') and self.zone:
            body['zone'] = self.zone

        return body