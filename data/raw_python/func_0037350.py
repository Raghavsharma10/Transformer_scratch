def _batch_get_archive_listing(self, archive_names):
        '''
        Batched version of :py:meth:`~MongoDBManager._get_archive_listing`

        Returns a list of full archive listings from an iterable of archive
        names

        .. note ::

            Invalid archive names will simply not be returned, so the response
            may not be the same length as the supplied `archive_names`.

        Parameters
        ----------

        archive_names : list

            List of archive names

        Returns
        -------

        archive_listings : list

            List of archive listings

        '''

        res = self.collection.find({'_id': {'$in': list(archive_names)}})

        if res is None:
            res = []

        return res