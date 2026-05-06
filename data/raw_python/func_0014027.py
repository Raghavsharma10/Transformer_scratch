def get_length(topics, yaml_info):
    '''
    Find the length (# of rows) in the created dataframe
    '''
    total = 0
    info = yaml_info['topics']
    for topic in topics:
        for t in info:
            if t['topic'] == topic:
                total = total + t['messages']
                break
    return total