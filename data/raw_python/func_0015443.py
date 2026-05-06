def get(self, key, get_cas=False):
        """
        Get a key from server.

        :param key: Key's name
        :type key: six.string_types
        :param get_cas: If true, return (value, cas), where cas is the new CAS value.
        :type get_cas: boolean
        :return: Returns a key data from server.
        :rtype: object
        """
        for server in self.servers:
            value, cas = server.get(key)
            if value is not None:
                if get_cas:
                    return value, cas
                else:
                    return value
        if get_cas:
            return None, None