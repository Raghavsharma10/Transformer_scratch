def restorenx(self, name, value, pttl=0):
        """
        Restore serialized dump of a key back into redis

        :param name: str     the name of the redis key
        :param value: redis RDB-like serialization
        :param pttl: milliseconds till key expires
        :return: Future()
        """
        return self.eval(lua_restorenx, 1, name, pttl, value)