def _rebuild_circle(self):
        """Updates the hash ring."""
        self._hashring = {}
        self._sorted_keys = []
        total_weight = 0
        for node in self._nodes:
            total_weight += self._weights.get(node, 1)

        for node in self._nodes:
            weight = self._weights.get(node, 1)

            ks = math.floor((40 * len(self._nodes) * weight) / total_weight)

            for i in xrange(0, int(ks)):
                k = md5_bytes('%s-%s-salt' % (node, i))

                for l in xrange(0, 4):
                    key = ((k[3 + l * 4] << 24) | (k[2 + l * 4] << 16) |
                           (k[1 + l * 4] << 8) | k[l * 4])
                    self._hashring[key] = node
                    self._sorted_keys.append(key)

        self._sorted_keys.sort()