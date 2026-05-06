def list(self, full=False):
        """List all of the bundles in the remote"""

        if self.is_api:
            return self._list_api(full=full)
        else:
            return self._list_fs(full=full)