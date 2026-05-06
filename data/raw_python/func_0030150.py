def measures(self):
        """Iterate over all measures"""
        from ambry.valuetype.core import ROLE

        return [c for c in self.columns if c.role == ROLE.MEASURE]