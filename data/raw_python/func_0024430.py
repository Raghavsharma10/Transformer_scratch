def _parse_msg_for_influxdb(self, msgs):
        '''
        >>> from logagg.forwarders import InfluxDBForwarder
        >>> idbf = InfluxDBForwarder('no_host', '8086', 'deadpool',
        ...                             'chimichanga', 'logs', 'collection')

        >>> valid_log = [{u'data': {u'_force_this_as_field': 'CXNS CNS nbkbsd',
        ...             u'a': 1,
        ...             u'b': 2,
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
        ...   u'type': u'metric'}]

        >>> pointvalues = idbf._parse_msg_for_influxdb(valid_log)
        >>> from pprint import pprint
        >>> pprint(pointvalues)
        [{'fields': {u'data._force_this_as_field': "'CXNS CNS nbkbsd'",
                     u'data.a': 1,
                     u'data.b': 2},
          'measurement': u'some_log',
          'tags': {u'data.msg': u'this is a dummy log',
                   u'error_tb': u'',
                   u'file': u'/var/log/sample.log',
                   u'formatter': u'logagg.formatters.basescript',
                   u'host': u'deepcompute',
                   u'level': u'info'},
          'time': u'2018-04-09T09:59:24.733945Z'}]

        >>> invalid_log = valid_log
        >>> invalid_log[0]['error'] = True
        >>> pointvalues = idbf._parse_msg_for_influxdb(invalid_log)
        >>> pprint(pointvalues)
        []

        >>> invalid_log = valid_log
        >>> invalid_log[0]['type'] = 'log'
        >>> pointvalues = idbf._parse_msg_for_influxdb(invalid_log)
        >>> pprint(pointvalues)
        []
        '''

        series = []

        for msg in msgs:
            if msg.get('error'):
                continue

            if msg.get('type').lower() == 'metric':
                time = msg.get('timestamp')
                measurement = msg.get('event')
                tags, fields = self._tag_and_field_maker(msg)
                pointvalues = {
                    "time": time,
                    "measurement": measurement,
                    "fields": fields,
                    "tags": tags}
                series.append(pointvalues)

        return series