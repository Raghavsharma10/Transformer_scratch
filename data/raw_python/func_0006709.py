def sync(self):
        """Sync this model with latest data on the saltant server.

        Note that in addition to returning the updated object, it also
        updates the existing object.

        Returns:
            :class:`saltant.models.user.User`:
                This user instance after syncing.
        """
        self = self.manager.get(username=self.username)

        return self