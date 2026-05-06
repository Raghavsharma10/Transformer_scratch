def split_segment_ids(segment_ids):
    """Given an array of strings of the form ifo:name and
    ifo:name:version, returns an array of tuples of the form (ifo,
    name, version) where version may be None"""
    
    def split_segment_id(segment_id):
        temp = segment_id.split(':')
        if len(temp) == 2:
            temp.append(None)
        elif temp[2] == '*':
            temp[2] = None
        else:
            temp[2] = int(temp[2])
            
        return temp

    return map(split_segment_id, segment_ids)