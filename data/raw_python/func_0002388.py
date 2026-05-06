def get_last(self):
        """
        Get the last migration batch.

        :rtype: list
        """
        query = self.table().where('batch', self.get_last_batch_number())

        return query.order_by('migration', 'desc').get()