def wait_until(time_label):
    '''
    Calculates the number of seconds that the process needs to sleep 
    '''
    if time_label == 'next_minute':
        gevent.sleep(60 - int(time.time()) % 60)
    elif time_label == 'next_hour':
        gevent.sleep(3600 - int(time.time()) % 3600)
    elif time_label == 'tomorrow':
        gevent.sleep(86400 - int(time.time()) % 86400)