def mongodb(line):
    '''
    >>> import pprint
    >>> input_line1 = '2017-08-17T07:56:33.489+0200 I REPL     [signalProcessingThread] shutting down replication subsystems'
    >>> output_line1 = mongodb(input_line1)
    >>> pprint.pprint(output_line1)
    {'data': {'component': 'REPL',
              'context': '[signalProcessingThread]',
              'message': 'shutting down replication subsystems',
              'severity': 'I',
              'timestamp': '2017-08-17T07:56:33.489+0200'},
     'timestamp': '2017-08-17T07:56:33.489+0200',
     'type': 'log'}

    >>> input_line2 = '2017-08-17T07:56:33.515+0200 W NETWORK  [initandlisten] No primary detected for set confsvr_repl1'
    >>> output_line2 = mongodb(input_line2)
    >>> pprint.pprint(output_line2)
    {'data': {'component': 'NETWORK',
              'context': '[initandlisten]',
              'message': 'No primary detected for set confsvr_repl1',
              'severity': 'W',
              'timestamp': '2017-08-17T07:56:33.515+0200'},
     'timestamp': '2017-08-17T07:56:33.515+0200',
     'type': 'log'}
    '''

    keys = ['timestamp', 'severity', 'component', 'context', 'message']
    values = re.split(r'\s+', line, maxsplit=4)
    mongodb_log = dict(zip(keys,values))

    return dict(
        timestamp=values[0],
        data=mongodb_log,
        type='log',
    )