def dimensions(self):
        """Iterate over the dimension columns, regardless of parent/child status

        """
        from ambry.valuetype.core import ROLE

        for c in self.columns:

            if c.role == ROLE.DIMENSION:
                yield c