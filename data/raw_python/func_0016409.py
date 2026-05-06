def present(self, path, timeout=0):
        """returns True if there is an entity at path"""

        ret, data = self.sendmess(MSG_PRESENCE, str2bytez(path),
                                  timeout=timeout)
        assert ret <= 0 and not data, (ret, data)
        if ret < 0:
            return False
        else:
            return True