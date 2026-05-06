def primary_measures(self):
        """Iterate over the primary columns, columns which do not have a parent

        Also sets the property partition_stats to the stats collection for the partition and column.
        """
        from ambry.valuetype.core import ROLE

        for c in self.columns:

            if not c.parent and c.role == ROLE.MEASURE:
                    yield c