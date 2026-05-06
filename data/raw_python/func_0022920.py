def del_key(self, ref):
        """
        Delete a key.

        (ref)
        Return None or LCDd response on error
        """
        if ref not in self.keys:
            response = self.request("client_del_key %s" % (ref))
            self.keys.remove(ref)
            if "success" in response:
                return None
            else:
                return response