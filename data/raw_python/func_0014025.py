def parse_series_args(topics, fields):
    '''Return which topics and which field keys need to be examined
    for plotting'''
    keys = {}
    for field in fields:
        for topic in topics:
            if field.startswith(topic):
                keys[field] = (topic, field[len(topic) + 1:])

    return keys