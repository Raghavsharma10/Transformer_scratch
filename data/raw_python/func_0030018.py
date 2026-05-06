def label(self):
        """"Return first child of the column that is marked as a label. Returns self if the column is a label"""

        if self.valuetype_class.is_label():
            return self

        for c in self.table.columns:
            if c.parent == self.name and  c.valuetype_class.is_label():
                return c

        return None