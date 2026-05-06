def ref(self, ref):
        """Get a reference pointed to by ``ref``.

        The most common will be branches and tags. For a branch, you must
        specify 'heads/branchname' and for a tag, 'tags/tagname'. Essentially,
        the system should return any reference you provide it in the namespace,
        including notes and stashes (provided they exist on the server).

        :param str ref: (required)
        :returns: :class:`Reference <github3.git.Reference>`
        """
        json = None
        if ref:
            url = self._build_url('git', 'refs', ref, base_url=self._api)
            json = self._json(self._get(url), 200)
        return Reference(json, self) if json else None