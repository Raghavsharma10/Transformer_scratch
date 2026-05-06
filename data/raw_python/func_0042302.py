def save(self):
        """Save this object to the database.  Behaves very similarly to
        whatever collection.save(document) would, ie. does upserts on _id
        presence.  If methods ``pre_save`` or ``post_save`` are defined, those
        are called.  If there is a spec document, then the document is
        validated against it after the ``pre_save`` hook but before the save."""
        if hasattr(self, 'pre_save'):
            self.pre_save()
        database, collection = self._collection_key.split('.')
        self.validate()
        _id = current()[database][collection].save(dict(self))
        if _id: self._id = _id
        if hasattr(self, 'post_save'):
            self.post_save()