def django(line):
    '''
    >>> import pprint
    >>> input_line1 = '[23/Aug/2017 11:35:25] INFO [app.middleware_log_req:50]View func called:{"exception": null,"processing_time": 0.00011801719665527344, "url": "<url>",host": "localhost", "user": "testing", "post_contents": "", "method": "POST" }'
    >>> output_line1 = django(input_line1)
    >>> pprint.pprint(output_line1)
    {'data': {'loglevel': 'INFO',
              'logname': '[app.middleware_log_req:50]',
              'message': 'View func called:{"exception": null,"processing_time": 0.00011801719665527344, "url": "<url>",host": "localhost", "user": "testing", "post_contents": "", "method": "POST" }',
              'timestamp': '2017-08-23T11:35:25'},
     'level': 'INFO',
     'timestamp': '2017-08-23T11:35:25'}

    >>> input_line2 = '[22/Sep/2017 06:32:15] INFO [app.function:6022] {"UUID": "c47f3530-9f5f-11e7-a559-917d011459f7", "timestamp":1506061932546, "misc": {"status": 200, "ready_state": 4, "end_time_ms": 1506061932546, "url": "/api/function?", "start_time_ms": 1506061932113, "response_length": 31, "status_message": "OK", "request_time_ms": 433}, "user": "root", "host_url": "localhost:8888", "message": "ajax success"}'
    >>> output_line2 = django(input_line2)
    >>> pprint.pprint(output_line2)
    {'data': {'loglevel': 'INFO',
              'logname': '[app.function:6022]',
              'message': {u'UUID': u'c47f3530-9f5f-11e7-a559-917d011459f7',
                          u'host_url': u'localhost:8888',
                          u'message': u'ajax success',
                          u'misc': {u'end_time_ms': 1506061932546L,
                                    u'ready_state': 4,
                                    u'request_time_ms': 433,
                                    u'response_length': 31,
                                    u'start_time_ms': 1506061932113L,
                                    u'status': 200,
                                    u'status_message': u'OK',
                                    u'url': u'/api/function?'},
                          u'timestamp': 1506061932546L,
                          u'user': u'root'},
              'timestamp': '2017-09-22T06:32:15'},
     'level': 'INFO',
     'timestamp': '2017-09-22T06:32:15'}

        Case2:
    [18/Sep/2017 05:40:36] ERROR [app.apps:78] failed to get the record, collection = Collection(Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True, serverselectiontimeoutms=3000), u'collection_cache'), u'function_dummy_version')
    Traceback (most recent call last):
      File "/usr/local/lib/python2.7/dist-packages/mongo_cache/mongocache.py", line 70, in __getitem__
    result = self.collection.find_one({"_id": key})
    OperationFailure: not authorized on collection_cache to execute command { find: "function", filter: { _id: "zydelig-cosine-20" }, limit: 1, singleBatch: true }
    '''
#TODO we need to handle case2 logs
    data = {}
    log = re.findall(r'^(\[\d+/\w+/\d+ \d+:\d+:\d+\].*)', line)
    if len(log) == 1:
        data['timestamp'] = datetime.datetime.strptime(re.findall(r'(\d+/\w+/\d+ \d+:\d+:\d+)',\
                log[0])[0],"%d/%b/%Y %H:%M:%S").isoformat()
        data['loglevel'] = re.findall('[A-Z]+', log[0])[1]
        data['logname'] = re.findall('\[\D+.\w+:\d+\]', log[0])[0]
        message = re.findall('\{.+\}', log[0])
        try:
            if len(message) > 0:
                message = json.loads(message[0])
            else:
                message = re.split(']', log[0])
                message = ''.join(message[2:])
        except ValueError:
            message = re.split(']', log[0])
            message = ''.join(message[2:])

        data['message'] = message

        return dict(
                timestamp=data['timestamp'],
                level=data['loglevel'],
                data=data,
            )
    else:
        return dict(
            timestamp=datetime.datetime.isoformat(datetime.datetime.utcnow()),
            data={raw:line}
        )