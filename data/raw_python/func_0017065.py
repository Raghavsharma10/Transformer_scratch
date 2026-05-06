def blob(self, sha):
        """Get the blob indicated by ``sha``.

        :param str sha: (required), sha of the blob
        :returns: :class:`Blob <github3.git.Blob>` if successful, otherwise
            None
        """
        url = self._build_url('git', 'blobs', sha, base_url=self._api)
        json = self._json(self._get(url), 200)
        return Blob(json) if json else None