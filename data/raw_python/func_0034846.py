def _pick_cluster_host(self, value):
        """Selects the Redis cluster host for the specified value.

        :param mixed value: The value to use when looking for the host
        :rtype: tredis.client._Connection

        """
        crc = crc16.crc16(self._encode_resp(value[1])) % HASH_SLOTS
        for host in self._cluster.keys():
            for slot in self._cluster[host].slots:
                if slot[0] <= crc <= slot[1]:
                    return self._cluster[host]
        LOGGER.debug('Host not found for %r, returning first connection',
                     value)
        host_keys = sorted(list(self._cluster.keys()))
        return self._cluster[host_keys[0]]