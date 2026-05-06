def elasticsearch(line):
    '''
    >>> import pprint
    >>> input_line = '[2017-08-30T06:27:19,158] [WARN ][o.e.m.j.JvmGcMonitorService] [Glsuj_2] [gc][296816] overhead, spent [1.2s] collecting in the last [1.3s]'
    >>> output_line = elasticsearch(input_line)
    >>> pprint.pprint(output_line)
    {'data': {'garbage_collector': 'gc',
              'gc_count': 296816.0,
              'level': 'WARN',
              'message': 'o.e.m.j.JvmGcMonitorService',
              'plugin': 'Glsuj_2',
              'query_time_ms': 1200.0,
              'resp_time_ms': 1300.0,
              'timestamp': '2017-08-30T06:27:19,158'},
     'event': 'o.e.m.j.JvmGcMonitorService',
     'level': 'WARN ',
     'timestamp': '2017-08-30T06:27:19,158',
     'type': 'metric'}

    Case 2:
    [2017-09-13T23:15:00,415][WARN ][o.e.i.e.Engine           ] [Glsuj_2] [filebeat-2017.09.09][3] failed engine [index]
    java.nio.file.FileSystemException: /home/user/elasticsearch/data/nodes/0/indices/jsVSO6f3Rl-wwBpQyNRCbQ/3/index/_0.fdx: Too many open files
            at sun.nio.fs.UnixException.translateToIOException(UnixException.java:91) ~[?:?]
    '''

    # TODO we need to handle case2 logs
    elasticsearch_log = line
    actuallog = re.findall(r'(\[\d+\-+\d+\d+\-+\d+\w+\d+:\d+:\d+,+\d\d\d+\].*)', elasticsearch_log)
    if len(actuallog) == 1:
        keys = ['timestamp','level','message','plugin','garbage_collector','gc_count','query_time_ms', 'resp_time_ms']
        values = re.findall(r'\[(.*?)\]', actuallog[0])
        for index, i in enumerate(values):
            if not isinstance(i, str):
                continue
            if len(re.findall(r'.*ms$', i)) > 0 and 'ms' in re.findall(r'.*ms$', i)[0]:
                num = re.split('ms', i)[0]
                values[index]  = float(num)
                continue
            if len(re.findall(r'.*s$', i)) > 0 and 's' in re.findall(r'.*s$', i)[0]:
                num = re.split('s', i)[0]
                values[index] = float(num) * 1000
                continue

        data = dict(zip(keys,values))
        if 'level' in data and data['level'][-1] == ' ':
            data['level'] = data['level'][:-1]
        if 'gc_count' in data:
            data['gc_count'] = float(data['gc_count'])
        event = data['message']
        level=values[1]
        timestamp=values[0]

        return dict(
                timestamp=timestamp,
                level=level,
                type='metric',
                data=data,
                event=event
        )

    else:
        return dict(
                timestamp=datetime.datetime.isoformat(datetime.datetime.now()),
                data={'raw': line}
        )