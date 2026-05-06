def _remove_redundancy(self, log):
        """Removes duplicate data from 'data' inside log dict and brings it
        out.

        >>> lc = LogCollector('file=/path/to/log_file.log:formatter=logagg.formatters.basescript', 30)

        >>> log = {'id' : 46846876, 'type' : 'log',
        ...         'data' : {'a' : 1, 'b' : 2, 'type' : 'metric'}}
        >>> lc._remove_redundancy(log)
        {'data': {'a': 1, 'b': 2}, 'type': 'metric', 'id': 46846876}
        """
        for key in log:
            if key in log and key in log['data']:
                log[key] = log['data'].pop(key)
        return log