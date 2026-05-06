def delete(self, dry=False, meta=None, index_fields=None):
        """
        Sets the objects "deleted" field to True and,
        current time to "deleted_at" fields then saves it to DB.


        Args:
            dry (bool): False. Do not execute the actual deletion.
            Just list what will be deleted as a result of relations.
            meta (dict): JSON serializable meta data for logging of save operation.
                {'lorem': 'ipsum', 'dolar': 5}
            index_fields (list): Tuple list for secondary indexing keys in riak (with 'bin' or 'int').
                bin is used for string fields, int is used for integer fields.
                [('lorem','bin'),('dolar','int')]
        Returns:
            Tuple. (results [], errors [])
        """
        from datetime import datetime
        # TODO: Make sure this works safely (like a sql transaction)
        if not dry:
            self.pre_delete()
        results, errors = self._delete_relations(dry)
        if not (dry or errors):
            self.deleted = True
            self.deleted_at = datetime.now()
            self.save(internal=True, meta=meta, index_fields=index_fields)
            self.post_delete()
            if settings.ENABLE_CACHING:
                cache.delete(self.key)
        return results, errors