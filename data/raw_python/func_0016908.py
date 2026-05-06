def iter_labels(self, number=-1, etag=None):
        """Iterate over the labels for every issue associated with this
        milestone.

        .. versionchanged:: 0.9

            Add etag parameter.

        :param int number: (optional), number of labels to return. Default: -1
            returns all available labels.
        :param str etag: (optional), ETag header from a previous response
        :returns: generator of :class:`Label <github3.issues.label.Label>`\ s
        """
        url = self._build_url('labels', base_url=self._api)
        return self._iter(int(number), url, Label, etag=etag)