def commit(self, collection, openSearcher=False, softCommit=False,
               waitSearcher=True, commit=True, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param bool openSearcher: If new searcher is to be opened
        :param bool softCommit: SoftCommit
        :param bool waitServer: Blocks until the new searcher is opened
        :param bool commit: Commit

        Sends a commit to a Solr collection.

        """
        comm = {
            'openSearcher': str(openSearcher).lower(),
            'softCommit': str(softCommit).lower(),
            'waitSearcher': str(waitSearcher).lower(),
            'commit': str(commit).lower()
        }

        self.logger.debug("Sending Commit to Collection {}".format(collection))
        try:
            resp, con_inf = self.transport.send_request(method='GET', endpoint='update', collection=collection,
                                                        params=comm, **kwargs)
        except Exception as e:
            raise
        self.logger.debug("Commit Successful, QTime is {}".format(resp['responseHeader']['QTime']))