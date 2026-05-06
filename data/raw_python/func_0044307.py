def _all_queue_names(self):
        """
        Return a list of all unique queue names in our config.

        :return: list of all queue names (str)
        :rtype: :std:term:`list`
        """
        queues = set()
        endpoints = self.config.get('endpoints')
        for e in endpoints:
            for q in endpoints[e]['queues']:
                queues.add(q)
        return sorted(queues)