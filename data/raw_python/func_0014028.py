def create_data_map(msgs_to_read):
    '''
    Create a data map for usage when parsing the bag
    '''
    dmap = {}
    for topic in msgs_to_read.keys():
        base_name = get_key_name(topic) + '__'
        fields = {}
        for f in msgs_to_read[topic]:
            key = (base_name + f).replace('.', '_')
            fields[f] = key
        dmap[topic] = fields
    return dmap