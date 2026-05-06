def remove(self, ref, cb=None):
        """Check in a bundle to the remote"""

        if self.is_api:
            return self._remove_api(ref, cb)
        else:
            return self._remove_fs(ref, cb)