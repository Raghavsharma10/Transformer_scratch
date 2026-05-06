def get_handlerecord_indices_for_key(self, key, list_of_entries):
        '''
        Finds the Handle entry indices of all entries that have a specific
        type.

        *Important:* It finds the Handle System indices! These are not
        the python indices of the list, so they can not be used for
        iteration.

        :param key: The key (Handle Record type)
        :param list_of_entries: A list of the existing entries in which to find
            the indices.
        :return: A list of strings, the indices of the entries of type "key" in
            the given handle record.
        '''

        LOGGER.debug('get_handlerecord_indices_for_key...')

        indices = []
        for entry in list_of_entries:
            if entry['type'] == key:
                indices.append(entry['index'])
        return indices