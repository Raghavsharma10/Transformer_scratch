def _get_archive_listing(self, archive_name):
        '''
        Return full document for ``{_id:'archive_name'}``

        .. note::

            MongoDB specific results - do not expose to user
        '''

        res = self.collection.find_one({'_id': archive_name})

        if res is None:
            raise KeyError

        return res