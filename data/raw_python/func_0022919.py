def add_key(self, ref, mode="shared"):
        """
        Add a key.

        (ref)
        Return key name or None on error
        """
        if ref not in self.keys:
            response = self.request("client_add_key %s -%s" % (ref, mode))
            if "success" not in response:
                return None
            self.keys.append(ref)
            return ref