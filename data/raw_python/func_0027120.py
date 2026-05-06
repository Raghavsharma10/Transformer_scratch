def row_dict(self, row):
        """returns dictionary version of row using keys from self.field_map"""
        d = {}
        for field_name,index in self.field_map.items():
            d[field_name] = self.field_value(row, field_name)
        return d