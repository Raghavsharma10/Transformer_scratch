def haproxy(line):
    #TODO Handle all message formats
    '''
    >>> import pprint
    >>> input_line1 = 'Apr 24 00:00:02 node haproxy[12298]: 1.1.1.1:48660 [24/Apr/2019:00:00:02.358] pre-staging~ pre-staging_doc/pre-staging_active 261/0/2/8/271 200 2406 - - ---- 4/4/0/1/0 0/0 {AAAAAA:AAAAA_AAAAA:AAAAA_AAAAA_AAAAA:300A||| user@mail.net:sdasdasdasdsdasAHDivsjd=|user@mail.net|2018} "GET /doc/api/get?call=apple HTTP/1.1"'
    >>> output_line1 = haproxy(input_line1)
    >>> pprint.pprint(output_line1)
    {'data': {'Tc': 2.0,
	      'Tq': 261.0,
	      'Tr': 8.0,
	      'Tw': 0.0,
	      '_api': '/doc/api/get?call=apple',
	      '_headers': ['AAAAAA:AAAAA_AAAAA:AAAAA_AAAAA_AAAAA:300A||| user@mail.net:sdasdasdasdsdasAHDivsjd=|user@mail.net|2018'],
	      'actconn': 4,
	      'backend': 'pre-staging_doc/pre-staging_active',
	      'backend_queue': 0,
	      'beconn': 1,
	      'bytes_read': 2406.0,
	      'client_port': '48660',
	      'client_server': '1.1.1.1',
	      'feconn': 4,
	      'front_end': 'pre-staging~',
	      'haproxy_server': 'node',
	      'method': 'GET',
	      'resp_time': 271.0,
	      'retries': 0,
	      'srv_conn': 0,
	      'srv_queue': 0,
	      'status': '200',
	      'timestamp': '2019-04-24T00:00:02.358000'},
     'event': 'haproxy_event',
     'timestamp': '2019-04-24T00:00:02.358000',
     'type': 'metric'}
    '''

    _line = line.strip().split()

    log = {}
    log['client_server'] = _line[5].split(':')[0].strip()
    log['client_port'] = _line[5].split(':')[1].strip()

    _timestamp = re.findall(r'\[(.*?)\]', _line[6])[0]
    log['timestamp'] = datetime.datetime.strptime(_timestamp, '%d/%b/%Y:%H:%M:%S.%f').isoformat()

    log['front_end'] = _line[7].strip()
    log['backend'] = _line[8].strip()

    log['Tq'] = float(_line[9].split('/')[0].strip())
    log['Tw'] = float(_line[9].split('/')[1].strip())
    log['Tc'] = float(_line[9].split('/')[2].strip())
    log['Tr'] = float(_line[9].split('/')[3].strip())
    log['resp_time'] = float(_line[9].split('/')[-1].strip())
    log['status'] = _line[10].strip()
    log['bytes_read'] = float(_line[11].strip())

    log['_headers'] = re.findall(r'{(.*)}', line)
    log['haproxy_server'] = _line[3].strip()

    log['method'] = _line[-3].strip('"').strip()
    log['_api'] = _line[-2].strip()

    log['retries'] = int(_line[15].split('/')[-1].strip())
    log['actconn'] = int(_line[15].split('/')[0].strip())
    log['feconn'] = int(_line[15].split('/')[1].strip())
    log['beconn'] = int(_line[15].split('/')[-2].strip())
    log['srv_conn'] = int(_line[15].split('/')[-3].strip())

    log['srv_queue'] = int(_line[16].split('/')[0].strip())
    log['backend_queue'] = int(_line[16].split('/')[1].strip())

    return dict(
        data=log,
        event='haproxy_event',
        timestamp=log.get('timestamp'),
        type='metric'
    )