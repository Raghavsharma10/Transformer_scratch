def from_db_value(self, value, expression, connection, context):
        '''Handle data loaded from database.'''
        if value is None:
            return value
        return self.parse_seconds(value)