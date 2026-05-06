def configure_owner(self, owner='www-data'):
        """Shortcut to set process owner data.

        :param str|unicode owner: Sets user and group. Default: ``www-data``.

        """
        if owner is not None:
            self.main_process.set_owner_params(uid=owner, gid=owner)

        return self