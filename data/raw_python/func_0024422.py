def assign_default_log_values(self, fpath, line, formatter):
        '''
        >>> lc = LogCollector('file=/path/to/log_file.log:formatter=logagg.formatters.basescript', 30)
        >>> from pprint import pprint

        >>> formatter = 'logagg.formatters.mongodb'
        >>> fpath = '/var/log/mongodb/mongodb.log'
        >>> line = 'some log line here'

        >>> default_log = lc.assign_default_log_values(fpath, line, formatter)
        >>> pprint(default_log) #doctest: +ELLIPSIS
        {'data': {},
         'error': False,
         'error_tb': '',
         'event': 'event',
         'file': '/var/log/mongodb/mongodb.log',
         'formatter': 'logagg.formatters.mongodb',
         'host': '...',
         'id': None,
         'level': 'debug',
         'raw': 'some log line here',
         'timestamp': '...',
         'type': 'log'}
        '''
        return dict(
            id=None,
            file=fpath,
            host=self.HOST,
            formatter=formatter,
            event='event',
            data={},
            raw=line,
            timestamp=datetime.datetime.utcnow().isoformat(),
            type='log',
            level='debug',
            error= False,
            error_tb='',
          )