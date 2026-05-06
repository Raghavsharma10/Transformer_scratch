def nodes_simple_info(self, params={}, **kwargs):
        """
        Return a dictionary of the nodes simple info that key is a column name,
        such as [{"http_address": "192.111.111.111", "name" : "test", ...}, ...]
        """
        h = ['name', 'pid', 'http_address', 'version', 'jdk', 'disk.total', 'disk.used_percent', 'heap.current',
             'heap.percent', 'ram.current', 'ram.percent', 'uptime', 'node.role']
        result = self.client.cat.nodes(v=True, h=h, **kwargs, params=params)
        result = [x.strip().split(' ') for x in result.split('\n')]
        # Clean up the space
        result.remove(result[-1])
        for i in range(len(result)):
            result[i] = list(filter(lambda x: x != '', result[i]))
        # Packing into the dictionary
        dicts = []
        for i in range(len(result) - 1):
            dict = {}
            for k, v in zip(result[0], result[i + 1]):
                dict[k] = v
            dicts.append(dict)

        logger.info('Acquire simple information of the nodes is done succeeded: %s' % len(dicts))
        return dicts