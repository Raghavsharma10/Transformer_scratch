def children(self):
        """"Return the table's other column that have this column as a parent, excluding labels"""
        for c in self.table.columns:
            if c.parent == self.name and  not c.valuetype_class.is_label():
                yield c