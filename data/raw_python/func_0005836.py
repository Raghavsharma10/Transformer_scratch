def get_owner(self, default=True):
        """Return (User ID, Group ID) tuple

        :param bool default: Whether to return default if not set.
        :rtype: tuple[int, int]
        """
        uid, gid = self.owner

        if not uid and default:
            uid = os.getuid()

        if not gid and default:
            gid = os.getgid()

        return uid, gid