def delete(self, **kwargs):
        """
        Deletes the document if it is saved in the collection.
        """
        if self.is_valid:
            if '_id' in self._document:
                to_delete = self.find_one({'_id': self._id})

                if to_delete:
                    before = self.before_delete()
                    if before:
                        return before

                    try:
                        self.delete_one({'_id': self._id})

                        self.after_delete()

                        return self._document
                    except PyMongoException as exc:
                        return PyMongoError(
                            error_message=exc.details.get(
                                'errmsg', exc.details.get(
                                    'err', 'PyMongoError.'
                                )
                            ),
                            operation='delete', collection=type(self).__name__,
                            document=self._document,
                        )
                else:
                    return DocumentNotFoundError(type(self).__name__, self._id)
            else:
                return UnidentifiedDocumentError(
                    type(self).__name__, self._document
                )