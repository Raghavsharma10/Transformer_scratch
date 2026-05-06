def get_multi(self, keys, get_cas=False):
        """
        Get multiple keys from server.

        :param keys: A list of keys to from server.
        :type keys: list
        :param get_cas: If get_cas is true, each value is (data, cas), with each result's CAS value.
        :type get_cas: boolean
        :return: A dict with all requested keys.
        :rtype: dict
        """
        servers = defaultdict(list)
        d = {}
        for key in keys:
            server_key = self._get_server(key)
            servers[server_key].append(key)
        for server, keys in servers.items():
            results = server.get_multi(keys)
            if not get_cas:
                # Remove CAS data
                for key, (value, cas) in results.items():
                    results[key] = value
            d.update(results)
        return d