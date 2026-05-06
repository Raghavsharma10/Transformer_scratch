def release_to_tag(self, release_id):
        """
        Shortcut to translate a release identifier to a tag name.

        :param release_id: A :attr:`Release.identifier` value (a string).
        :returns: A tag name (a string).
        :raises: :exc:`~exceptions.TypeError` when :attr:`release_scheme` isn't
                 'tags'.
        """
        self.ensure_release_scheme('tags')
        return self.releases[release_id].revision.tag