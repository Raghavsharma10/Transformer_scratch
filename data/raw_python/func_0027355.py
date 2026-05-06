def is_10320LOC_empty(self, handle, handlerecord_json=None):
        '''
        Checks if there is a 10320/LOC entry in the handle record.
        *Note:* In the unlikely case that there is a 10320/LOC entry, but it does
        not contain any locations, it is treated as if there was none.

        :param handle: The handle.
        :param handlerecord_json: Optional. The content of the response of a
            GET request for the handle as a dict. Avoids another GET request.
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :return: True if the record contains NO 10320/LOC entry; False if it
            does contain one.
        '''
        LOGGER.debug('is_10320LOC_empty...')

        handlerecord_json = self.__get_handle_record_if_necessary(handle, handlerecord_json)
        if handlerecord_json is None:
            raise HandleNotFoundException(handle=handle)
        list_of_entries = handlerecord_json['values']

        num_entries = 0
        num_URL = 0
        for entry in list_of_entries:
            if entry['type'] == '10320/LOC':
                num_entries += 1
                xmlroot = ET.fromstring(entry['data']['value'])
                list_of_locations = xmlroot.findall('location')
                for item in list_of_locations:
                    if item.get('href') is not None:
                        num_URL += 1
        if num_entries == 0:
            return True
        else:
            if num_URL == 0:
                return True
            else:
                return False