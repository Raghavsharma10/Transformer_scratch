def nginx_access(line):
    '''
    >>> import pprint
    >>> input_line1 = '{ \
                    "remote_addr": "127.0.0.1","remote_user": "-","timestamp": "1515144699.201", \
                    "request": "GET / HTTP/1.1","status": "200","request_time": "0.000", \
                    "body_bytes_sent": "396","http_referer": "-","http_user_agent": "python-requests/2.18.4", \
                    "http_x_forwarded_for": "-","upstream_response_time": "-" \
                        }'
    >>> output_line1 = nginx_access(input_line1)
    >>> pprint.pprint(output_line1)
    {'data': {u'body_bytes_sent': 396.0,
              u'http_referer': u'-',
              u'http_user_agent': u'python-requests/2.18.4',
              u'http_x_forwarded_for': u'-',
              u'remote_addr': u'127.0.0.1',
              u'remote_user': u'-',
              u'request': u'GET / HTTP/1.1',
              u'request_time': 0.0,
              u'status': u'200',
              u'timestamp': '2018-01-05T09:31:39.201000',
              u'upstream_response_time': 0.0},
     'event': 'nginx_event',
     'timestamp': '2018-01-05T09:31:39.201000',
     'type': 'metric'}

    >>> input_line2 = '{ \
                    "remote_addr": "192.158.0.51","remote_user": "-","timestamp": "1515143686.415", \
                    "request": "POST /mpub?topic=heartbeat HTTP/1.1","status": "404","request_time": "0.000", \
                    "body_bytes_sent": "152","http_referer": "-","http_user_agent": "python-requests/2.18.4", \
                    "http_x_forwarded_for": "-","upstream_response_time": "-" \
                       }'
    >>> output_line2 = nginx_access(input_line2)
    >>> pprint.pprint(output_line2)
    {'data': {u'body_bytes_sent': 152.0,
              u'http_referer': u'-',
              u'http_user_agent': u'python-requests/2.18.4',
              u'http_x_forwarded_for': u'-',
              u'remote_addr': u'192.158.0.51',
              u'remote_user': u'-',
              u'request': u'POST /mpub?topic=heartbeat HTTP/1.1',
              u'request_time': 0.0,
              u'status': u'404',
              u'timestamp': '2018-01-05T09:14:46.415000',
              u'upstream_response_time': 0.0},
     'event': 'nginx_event',
     'timestamp': '2018-01-05T09:14:46.415000',
     'type': 'metric'}
    '''
#TODO Handle nginx error logs
    log = json.loads(line)
    timestamp_iso = datetime.datetime.utcfromtimestamp(float(log['timestamp'])).isoformat()
    log.update({'timestamp':timestamp_iso})
    if '-' in log.get('upstream_response_time'):
        log['upstream_response_time'] = 0.0
    log['body_bytes_sent'] = float(log['body_bytes_sent'])
    log['request_time'] = float(log['request_time'])
    log['upstream_response_time'] = float(log['upstream_response_time'])

    return dict(
        timestamp=log.get('timestamp',' '),
        data=log,
        type='metric',
        event='nginx_event',
    )