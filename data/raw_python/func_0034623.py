def push(self, my_dict, key, element):
        ''' Push an element onto an array that may not have been defined in
        the dict '''
        group_info = my_dict.setdefault(key, [])
        if isinstance(group_info, dict):
            host_list = group_info.setdefault('hosts', [])
            host_list.append(element)
        else:
            group_info.append(element)