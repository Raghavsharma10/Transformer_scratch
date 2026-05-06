def get_msg_info(yaml_info, topics, parse_header=True):
    '''
    Get info from all of the messages about what they contain
    and will be added to the dataframe
    '''
    topic_info = yaml_info['topics']
    msgs = {}
    classes = {}

    for topic in topics:
        base_key = get_key_name(topic)
        msg_paths = []
        msg_types = {}
        for info in topic_info:
            if info['topic'] == topic:
                msg_class = get_message_class(info['type'])
                if msg_class is None:
                    warnings.warn(
                        'Could not find types for ' + topic + ' skpping ')
                else:
                    (msg_paths, msg_types) = get_base_fields(msg_class(), "",
                                                             parse_header)
                msgs[topic] = msg_paths
                classes[topic] = msg_types
    return (msgs, classes)