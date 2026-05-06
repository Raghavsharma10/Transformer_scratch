def checkout(self, ref, cb=None):
        """Checkout a bundle from the remote. Returns a file-like object"""
        if self.is_api:
            return self._checkout_api(ref, cb=cb)
        else:
            return self._checkout_fs(ref, cb=cb)