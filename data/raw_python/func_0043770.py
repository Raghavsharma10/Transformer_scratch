def _walk(self, path_to_root, record_dict):

        '''
            a helper method for finding the record endpoint from a path to root

        :param path_to_root: string with dot path to root from
        :param record_dict:
        :return: list, dict, string, number, or boolean at path to root
        '''

    # split path to root into segments
        item_pattern = re.compile('\d+\\]')
        dot_pattern = re.compile('\\.|\\[')
        path_segments = dot_pattern.split(path_to_root)

    # construct empty fields
        record_endpoints = []

    # determine starting position
        if not path_segments[0]:
            path_segments.pop(0)
        
    # define internal recursive function
        def _walk_int(path_segments, record_dict):
            record_endpoint = record_dict
            for i in range(0, len(path_segments)):
                if item_pattern.match(path_segments[i]):
                    for j in range(0, len(record_endpoint)):
                        if len(path_segments) == 2:
                            record_endpoints.append(record_endpoint[j])
                        else:
                            stop_chain = False
                            for x in range(0, i):
                                if item_pattern.match(path_segments[x]):
                                    stop_chain = True
                            if not stop_chain:
                                shortened_segments = []
                                for z in range(i + 1, len(path_segments)):
                                    shortened_segments.append(path_segments[z])
                                _walk_int(shortened_segments, record_endpoint[j])
                else:
                    stop_chain = False
                    for y in range(0, i):
                        if item_pattern.match(path_segments[y]):
                            stop_chain = True
                    if not stop_chain:
                        if len(path_segments) == i + 1:
                            record_endpoints.append(record_endpoint[path_segments[i]])
                        else:
                            record_endpoint = record_endpoint[path_segments[i]]

    # conduct recursive walk
        _walk_int(path_segments, record_dict)

        return record_endpoints