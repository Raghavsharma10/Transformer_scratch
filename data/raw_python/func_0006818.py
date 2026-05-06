def primary_keys(self):
        """Returns the primary keys referenced in user supplied model"""
        res = []
        for column in self.column_definitions:
            if 'primary_key' in column.keys():
                tmp = column.get('primary_key', None)
                res.append(column['name']) if tmp else False
        return res