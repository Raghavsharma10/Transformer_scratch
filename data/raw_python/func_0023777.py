def to_dict(self):
        """
        Return a dict that can be serialised to JSON and sent to UpCloud's API.
        """
        return dict(
            (attr, getattr(self, attr))
            for attr in self.ATTRIBUTES
            if hasattr(self, attr)
        )