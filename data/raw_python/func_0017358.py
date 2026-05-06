def transform(self, attrs):
        """Perform all actions on a given attribute dict."""
        self.collect(attrs)
        self.add_missing_implementations()
        self.fill_attrs(attrs)