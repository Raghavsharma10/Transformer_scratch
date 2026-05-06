def delete(self):
        """
        Delete the pivot model record from the database.

        :rtype: int
        """
        query = self._get_delete_query()

        query.where(self._morph_type, self._morph_class)

        return query.delete()