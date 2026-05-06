def update(self, **kwargs):
        """
        Updates the document with the given _id saved in the collection if it
        is valid.
        Returns errors otherwise.
        """
        if self.is_valid:
            if '_id' in self._document:
                to_update = self.find_one({'_id': self._id})

                if to_update:
                    before = self.before_update(old=to_update)
                    if before:
                        return before

                    try:
                        self.replace_one({'_id': self._id}, self._document)

                        self.after_update(old=to_update)

                        return self._document
                    except PyMongoException as exc:
                        return PyMongoError(
                            error_message=exc.details.get(
                                'errmsg', exc.details.get(
                                    'err', 'PyMongoError.'
                                )
                            ),
                            operation='update', collection=type(self).__name__,
                            document=self._document,
                        )
                else:
                    return DocumentNotFoundError(type(self).__name__, self._id)
            else:
                return UnidentifiedDocumentError(
                    type(self).__name__, self._document
                )

        return self._errors