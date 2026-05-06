def remote(self):
        """
        Return the remote for this partition

        :return:

        """
        from ambry.exc import NotFoundError

        ds = self.dataset

        if 'remote_name' not in ds.data:
            raise NotFoundError('Could not determine remote for partition: {}'.format(self.identity.fqname))

        return self._bundle.library.remote(ds.data['remote_name'])