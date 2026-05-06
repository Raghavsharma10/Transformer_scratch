def __remove_URL_from_10320LOC(self, url, list_of_entries, handle):
        '''
        Remove an URL from the handle record's "10320/LOC" entry.
        If it exists several times in the entry, all occurences are removed.
        If the URL is not present, nothing happens.
        If after removing, there is no more URLs in the entry, the entry is
            removed.

        :param url: The URL to be removed.
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

        num_removed = 0
        if len(python_indices) > 0:

            if len(python_indices) > 1:
                msg = str(len(python_indices)) + ' entries of type "10320/LOC".'
                raise BrokenHandleRecordException(handle=handle, msg=msg)

            for index in python_indices:
                entry = list_of_entries.pop(index)
                xmlroot = ET.fromstring(entry['data']['value'])
                all_URL_elements = xmlroot.findall('location')
                for element in all_URL_elements:
                    if element.get('href') == url:
                        LOGGER.debug('__remove_URL_from_10320LOC: Removing URL ' + url + '.')
                        num_removed += 1
                        xmlroot.remove(element)
                remaining_URL_elements = xmlroot.findall('location')
                if len(remaining_URL_elements) == 0:
                    LOGGER.debug("__remove_URL_from_10320LOC: All URLs removed.")
                    # TODO FIXME: If we start adapting the Handle Record by
                    # index (instead of overwriting the entire one), be careful
                    # to delete the ones that became empty!
                else:
                    entry['data']['value'] = ET.tostring(xmlroot, encoding=encoding_value)
                    LOGGER.debug('__remove_URL_from_10320LOC: ' + str(len(remaining_URL_elements)) + ' URLs' + \
                        ' left after removal operation.')
                    list_of_entries.append(entry)
        if num_removed == 0:
            LOGGER.debug('__remove_URL_from_10320LOC: No URLs removed.')
        else:
            message = '__remove_URL_from_10320LOC: The URL "' + url + '" was removed '\
            + str(num_removed) + ' times.'
            message = message.replace('1 times', 'once')
            LOGGER.debug(message)