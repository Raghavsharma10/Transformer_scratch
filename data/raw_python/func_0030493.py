def index_one(self, instance, force=False):
        """ Indexes exactly one object of the Ambry system.

        Args:
            instance (any): instance to index.
            force (boolean): if True replace document in the index.

        Returns:
            boolean: True if document added to index, False if document already exists in the index.
        """
        if not self.is_indexed(instance) and not force:
            doc = self._as_document(instance)
            self._index_document(doc, force=force)
            logger.debug('{} indexed as\n {}'.format(instance.__class__, pformat(doc)))
            return True

        logger.debug('{} already indexed.'.format(instance.__class__))
        return False