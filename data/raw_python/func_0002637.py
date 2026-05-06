def insert(self, **kwargs):
        """
        Saves the Document to the database if it is valid.
        Returns errors otherwise.
        """
        if self.is_valid:
            before = self.before_insert()
            if before:
                return before

            try:
                self._document['_id'] = self.insert_one(self._document)

                self.after_insert()

                return self._document
            except PyMongoException as exc:
                return PyMongoError(
                    error_message=exc.details.get(
                        'errmsg', exc.details.get('err', 'PyMongoError.')
                    ),
                    operation='insert', collection=type(self).__name__,
                    document=self._document,
                )

        return self._errors