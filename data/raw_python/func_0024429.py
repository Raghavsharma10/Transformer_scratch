def _tag_and_field_maker(self, event):
        '''
        >>> idbf = InfluxDBForwarder('no_host', '8086', 'deadpool',
        ...                             'chimichanga', 'logs', 'collection')
        >>> log = {u'data': {u'_': {u'file': u'log.py',
        ...                         u'fn': u'start',
        ...                         u'ln': 8,
        ...                         u'name': u'__main__'},
        ...             u'a': 1,
        ...             u'b': 2,
        ...             u'__ignore_this': 'some_string',
        ...             u'msg': u'this is a dummy log'},
        ...   u'error': False,
        ...   u'error_tb': u'',
        ...   u'event': u'some_log',
        ...   u'file': u'/var/log/sample.log',
        ...   u'formatter': u'logagg.formatters.basescript',
        ...   u'host': u'deepcompute',
        ...   u'id': u'20180409T095924_aec36d313bdc11e89da654e1ad04f45e',
        ...   u'level': u'info',
        ...   u'raw': u'{...}',
        ...   u'timestamp': u'2018-04-09T09:59:24.733945Z',
        ...   u'type': u'metric'}

        >>> tags, fields = idbf._tag_and_field_maker(log)
        >>> from pprint import pprint
        >>> pprint(tags)
        {u'data.msg': u'this is a dummy log',
         u'error_tb': u'',
         u'file': u'/var/log/sample.log',
         u'formatter': u'logagg.formatters.basescript',
         u'host': u'deepcompute',
         u'level': u'info'}
        >>> pprint(fields)
        {u'data._': "{u'ln': 8, u'fn': u'start', u'file': u'log.py', u'name': u'__main__'}",
         u'data.a': 1,
         u'data.b': 2}

        '''
        data = event.pop('data')
        data = flatten_dict({'data': data})

        t = dict((k, event[k]) for k in event if k not in self.EXCLUDE_TAGS)
        f = dict()

        for k in data:
            v = data[k]

            if is_number(v) or isinstance(v, MarkValue):
                f[k] = v
            else:
                #if v.startswith('_'): f[k] = eval(v.split('_', 1)[1])
                t[k] = v

        return t, f