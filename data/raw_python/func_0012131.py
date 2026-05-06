def assign(self, attrs):
        """Merge new attributes
        """
        for k, v in attrs.items():
            setattr(self, k, v)