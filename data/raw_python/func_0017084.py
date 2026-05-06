def create_ref(self, ref, sha):
        """Create a reference in this repository.

        :param str ref: (required), fully qualified name of the reference,
            e.g. ``refs/heads/master``. If it doesn't start with ``refs`` and
            contain at least two slashes, GitHub's API will reject it.
        :param str sha: (required), SHA1 value to set the reference to
        :returns: :class:`Reference <github3.git.Reference>` if successful
            else None
        """
        json = None
        if ref and ref.count('/') >= 2 and sha:
            data = {'ref': ref, 'sha': sha}
            url = self._build_url('git', 'refs', base_url=self._api)
            json = self._json(self._post(url, data=data), 201)
        return Reference(json, self) if json else None