def get_value_from_handle(self, handle, key, handlerecord_json=None):
        '''
        Retrieve a single value from a single Handle. If several entries with
        this key exist, the methods returns the first one. If the handle
        does not exist, the method will raise a HandleNotFoundException.

        :param handle: The handle to take the value from.
        :param key: The key.
        :return: A string containing the value or None if the Handle record
         does not contain the key.
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        '''
        LOGGER.debug('get_value_from_handle...')

        handlerecord_json = self.__get_handle_record_if_necessary(handle, handlerecord_json)
        if handlerecord_json is None:
            raise HandleNotFoundException(handle=handle)
        list_of_entries = handlerecord_json['values']

        indices = []
        for i in xrange(len(list_of_entries)):
            if list_of_entries[i]['type'] == key:
                indices.append(i)

        if len(indices) == 0:
            return None
        else:
            if len(indices) > 1:
                LOGGER.debug('get_value_from_handle: The handle ' + handle + \
                    ' contains several entries of type "' + key + \
                    '". Only the first one is returned.')
            return list_of_entries[indices[0]]['data']['value']