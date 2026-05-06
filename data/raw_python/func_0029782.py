def primary_dimensions(self):
        """Iterate over the primary dimension columns, columns which do not have a parent

        """
        from ambry.valuetype.core import ROLE

        for c in self.columns:

            if not c.parent and c.role == ROLE.DIMENSION:
                    yield c