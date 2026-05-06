def validate_log_format(self, log):
        '''
        >>> lc = LogCollector('file=/path/to/file.log:formatter=logagg.formatters.basescript', 30)

        >>> incomplete_log = {'data' : {'x' : 1, 'y' : 2},
        ...                     'raw' : 'Not all keys present'}
        >>> lc.validate_log_format(incomplete_log)
        'failed'

        >>> redundant_log = {'one_invalid_key' : 'Extra information',
        ...  'data': {'x' : 1, 'y' : 2},
        ...  'error': False,
        ...  'error_tb': '',
        ...  'event': 'event',
        ...  'file': '/path/to/file.log',
        ...  'formatter': 'logagg.formatters.mongodb',
        ...  'host': 'deepcompute-ThinkPad-E470',
        ...  'id': '0112358',
        ...  'level': 'debug',
        ...  'raw': 'some log line here',
        ...  'timestamp': '2018-04-07T14:06:17.404818',
        ...  'type': 'log'}
        >>> lc.validate_log_format(redundant_log)
        'failed'

        >>> correct_log = {'data': {'x' : 1, 'y' : 2},
        ...  'error': False,
        ...  'error_tb': '',
        ...  'event': 'event',
        ...  'file': '/path/to/file.log',
        ...  'formatter': 'logagg.formatters.mongodb',
        ...  'host': 'deepcompute-ThinkPad-E470',
        ...  'id': '0112358',
        ...  'level': 'debug',
        ...  'raw': 'some log line here',
        ...  'timestamp': '2018-04-07T14:06:17.404818',
        ...  'type': 'log'}
        >>> lc.validate_log_format(correct_log)
        'passed'
        '''

        keys_in_log = set(log)
        keys_in_log_structure = set(self.LOG_STRUCTURE)
        try:
            assert (keys_in_log == keys_in_log_structure)
        except AssertionError as e:
            self.log.warning('formatted_log_structure_rejected' ,
                                key_not_found = list(keys_in_log_structure-keys_in_log),
                                extra_keys_found = list(keys_in_log-keys_in_log_structure),
                                num_logs=1,
                                type='metric')
            return 'failed'

        for key in log:
            try:
                assert isinstance(log[key], self.LOG_STRUCTURE[key])
            except AssertionError as e:
                self.log.warning('formatted_log_structure_rejected' ,
                                    key_datatype_not_matched = key,
                                    datatype_expected = type(self.LOG_STRUCTURE[key]),
                                    datatype_got = type(log[key]),
                                    num_logs=1,
                                    type='metric')
                return 'failed'

        return 'passed'