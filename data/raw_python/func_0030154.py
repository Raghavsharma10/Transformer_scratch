def label(self):
        """"Return first child that of the column that is marked as a label"""
        for c in self.table.columns:
            if c.parent == self.name and 'label' in c.valuetype:
                return PartitionColumn(c, self._partition)