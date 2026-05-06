def get_message_data(msg, key):
    '''get the datapoint from the dot delimited message field key
    e.g. translation.x looks up translation than x and returns the value found
    in x'''
    data = msg
    paths = key.split('.')
    for i in paths:
        data = getattr(data, i)
    return data