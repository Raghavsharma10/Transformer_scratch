def __exchange_URL_in_13020loc(self, oldurl, newurl, list_of_entries, handle):
        '''
        Exchange every occurrence of oldurl against newurl in a 10320/LOC entry.
            This does not change the ids or other xml attributes of the
            <location> element.

        :param oldurl: The URL that will be overwritten.
        :param newurl: The URL to write into the entry.
        :param list_of_entries: A list of the existing entries (to find and
            remove the correct one).
        :param handle: Only for the exception message.
        :raise: GenericHandleError: If several 10320/LOC exist (unlikely).
        '''

        # Find existing 10320/LOC entries
        python_indices = self.__get_python_indices_for_key(
            '10320/LOC',
            list_of_entries
        )

        num_exchanged = 0
        if len(python_indices) > 0:

            if len(python_indices) > 1:
                msg = str(len(python_indices)) + ' entries of type "10320/LOC".'
                raise BrokenHandleRecordException(handle=handle, msg=msg)

            for index in python_indices:
                entry = list_of_entries.pop(index)
                xmlroot = ET.fromstring(entry['data']['value'])
                all_URL_elements = xmlroot.findall('location')
                for element in all_URL_elements:
                    if element.get('href') == oldurl:
                        LOGGER.debug('__exchange_URL_in_13020loc: Exchanging URL ' + oldurl + ' from 10320/LOC.')
                        num_exchanged += 1
                        element.set('href', newurl)
                entry['data']['value'] = ET.tostring(xmlroot, encoding=encoding_value)
                list_of_entries.append(entry)

        if num_exchanged == 0:
            LOGGER.debug('__exchange_URL_in_13020loc: No URLs exchanged.')
        else:
            message = '__exchange_URL_in_13020loc: The URL "' + oldurl + '" was exchanged ' + str(num_exchanged) + \
            ' times against the new url "' + newurl + '" in 10320/LOC.'
            message = message.replace('1 times', 'once')
            LOGGER.debug(message)