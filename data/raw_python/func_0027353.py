def retrieve_handle_record(self, handle, handlerecord_json=None):
        '''
        Retrieve a handle record from the Handle server as a dict. If there
        is several entries of the same type, only the first one is
        returned. Values of complex types (such as HS_ADMIN) are
        transformed to strings.

        :param handle: The handle whose record to retrieve.
        :param handlerecord_json: Optional. If the handlerecord has already
            been retrieved from the server, it can be reused.
        :return: A dict where the keys are keys from the Handle record (except
            for hidden entries) and every value is a string. The result will be
            None if the Handle does not exist.
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        '''
        LOGGER.debug('retrieve_handle_record...')

        handlerecord_json = self.__get_handle_record_if_necessary(handle, handlerecord_json)
        if handlerecord_json is None:
            return None  # Instead of HandleNotFoundException!
        list_of_entries = handlerecord_json['values']

        record_as_dict = {}
        for entry in list_of_entries:
            key = entry['type']
            if not key in record_as_dict.keys():
                record_as_dict[key] = str(entry['data']['value'])
        return record_as_dict