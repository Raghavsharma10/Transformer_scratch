def __get_python_indices_for_key(self, key, list_of_entries):
        '''
        Finds the indices of all entries that have a specific type. Important:
            This method finds the python indices of the list of entries! These
            are not the Handle System index values!

        :param key: The key (Handle Record type)
        :param list_of_entries: A list of the existing entries in which to find
            the indices.
        :return: A list of integers, the indices of the entries of type "key"
            in the given list.
        '''
        indices = []
        for i in xrange(len(list_of_entries)):
            if list_of_entries[i]['type'] == key:
                indices.append(i)
        return indices