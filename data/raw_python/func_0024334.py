def basescript(line):
    '''
    >>> import pprint
    >>> input_line = '{"level": "warning", "timestamp": "2018-02-07T06:37:00.297610Z", "event": "exited via keyboard interrupt", "type": "log", "id": "20180207T063700_4d03fe800bd111e89ecb96000007bc65", "_": {"ln": 58, "file": "/usr/local/lib/python2.7/dist-packages/basescript/basescript.py", "name": "basescript.basescript", "fn": "start"}}'
    >>> output_line1 = basescript(input_line)
    >>> pprint.pprint(output_line1)
    {'data': {u'_': {u'file': u'/usr/local/lib/python2.7/dist-packages/basescript/basescript.py',
                     u'fn': u'start',
                     u'ln': 58,
                     u'name': u'basescript.basescript'},
              u'event': u'exited via keyboard interrupt',
              u'id': u'20180207T063700_4d03fe800bd111e89ecb96000007bc65',
              u'level': u'warning',
              u'timestamp': u'2018-02-07T06:37:00.297610Z',
              u'type': u'log'},
     'event': u'exited via keyboard interrupt',
     'id': u'20180207T063700_4d03fe800bd111e89ecb96000007bc65',
     'level': u'warning',
     'timestamp': u'2018-02-07T06:37:00.297610Z',
     'type': u'log'}
    '''

    log = json.loads(line)

    return dict(
        timestamp=log['timestamp'],
        data=log,
        id=log['id'],
        type=log['type'],
        level=log['level'],
        event=log['event']
    )