def new_partition(self, table, **kwargs):
        """ Creates new partition and returns it.

        Args:
            table (orm.Table):

        Returns:
            orm.Partition
        """

        from . import Partition

        # Create the basic partition record, with a sequence ID.

        if isinstance(table, string_types):
            table = self.table(table)

        if 'sequence_id' in kwargs:
            sequence_id = kwargs['sequence_id']
            del kwargs['sequence_id']
        else:
            sequence_id = self._database.next_sequence_id(Dataset, self.vid, Partition)

        p = Partition(
            t_vid=table.vid,
            table_name=table.name,
            sequence_id=sequence_id,
            dataset=self,
            d_vid=self.vid,
            **kwargs
        )


        p.update_id()

        return p