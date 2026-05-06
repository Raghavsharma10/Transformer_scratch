def get_db_prep_value(self, value, connection, prepared=False):
        """Convert JSON object to a string"""
        if isinstance(value, basestring):
            return value
        return json.dumps(value, **self.dump_kwargs)