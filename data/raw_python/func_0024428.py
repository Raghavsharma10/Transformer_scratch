def _parse_msg_for_mongodb(self, msgs):
        '''
        >>> mdbf = MongoDBForwarder('no_host', '27017', 'deadpool',
        ...                             'chimichanga', 'logs', 'collection')
        >>> log = [{u'data': {u'_': {u'file': u'log.py',
        ...                    u'fn': u'start',
        ...                    u'ln': 8,
        ...                    u'name': u'__main__'},
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

        >>> records = mdbf._parse_msg_for_mongodb(log)
        >>> from pprint import pprint
        >>> pprint(records)
        [{'_id': u'20180409T095924_aec36d313bdc11e89da654e1ad04f45e',
          u'data': {u'_': {u'file': u'log.py',
                           u'fn': u'start',
                           u'ln': 8,
                           u'name': u'__main__'},
                    u'a': 1,
                    u'b': 2,
                    u'msg': u'this is a dummy log'},
          u'error': False,
          u'error_tb': u'',
          u'event': u'some_log',
          u'file': u'/var/log/sample.log',
          u'formatter': u'logagg.formatters.basescript',
          u'host': u'deepcompute',
          u'level': u'info',
          u'raw': u'{...}',
          u'timestamp': u'2018-04-09T09:59:24.733945Z',
          u'type': u'metric'}]
        '''
        msgs_list = []
        for msg in msgs:
            try:
                msg['_id'] = msg.pop('id')
            except KeyError:
                self.log.exception('collector_failure_id_not_found', log=msg)
            msgs_list.append(msg)
        return msgs_list