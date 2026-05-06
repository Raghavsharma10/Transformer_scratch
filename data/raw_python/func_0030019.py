def geoid(self):
        """"Return first child of the column, or self that is marked as a geographic identifier"""

        if self.valuetype_class.is_geoid():
            return self

        for c in self.table.columns:
            if c.parent == self.name and  c.valuetype_class.is_geoid():
                return c