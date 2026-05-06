def elasticsearch_ispartial_log(line):
    '''
    >>> line1 = '  [2018-04-03T00:22:38,048][DEBUG][o.e.c.u.c.QueueResizingEsThreadPoolExecutor] [search17/search]: there were [2000] tasks in [809ms], avg task time [28.4micros], EWMA task execution [790nanos], [35165.36 tasks/s], optimal queue is [35165], current capacity [1000]'
    >>> line2 = '  org.elasticsearch.ResourceAlreadyExistsException: index [media_corpus_refresh/6_3sRAMsRr2r63J6gbOjQw] already exists'
    >>> line3 = '   at org.elasticsearch.cluster.metadata.MetaDataCreateIndexService.validateIndexName(MetaDataCreateIndexService.java:151) ~[elasticsearch-6.2.0.jar:6.2.0]'
    >>> elasticsearch_ispartial_log(line1)
    False
    >>> elasticsearch_ispartial_log(line2)
    True
    >>> elasticsearch_ispartial_log(line3)
    True
    '''
    match_result = []

    for p in LOG_BEGIN_PATTERN:
        if re.match(p, line) != None:
            return False
    return True