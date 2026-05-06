def ensure_exists(self):
        """
        Make sure the local repository exists.

        :raises: :exc:`~exceptions.ValueError` when the
                 local repository doesn't exist yet.
        """
        if not self.exists:
            msg = "The local %s repository %s doesn't exist!"
            raise ValueError(msg % (self.friendly_name, format_path(self.local)))