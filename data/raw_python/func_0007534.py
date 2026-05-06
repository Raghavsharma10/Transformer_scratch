def current(self, fields=None):
        """Returns dict of current values for all tracked fields"""
        if fields is None:
            fields = self.fields

        return dict((f, self.get_field_value(f)) for f in fields)