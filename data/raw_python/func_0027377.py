def __add_URL_to_10320LOC(self, url, list_of_entries, handle=None, weight=None, http_role=None, **kvpairs):
        '''
        Add a url to the handle record's "10320/LOC" entry.
            If no 10320/LOC entry exists, a new one is created (using the
            default "chooseby" attribute, if configured).
            If the URL is already present, it is not added again, but
            the attributes (e.g. weight) are updated/added.
            If the existing 10320/LOC entry is mal-formed, an exception will be
            thrown (xml.etree.ElementTree.ParseError)
            Note: In the unlikely case that several "10320/LOC" entries exist,
            an exception is raised.

        :param url: The URL to be added.
        :param list_of_entries: A list of the existing entries (to find and
            adapt the correct one).
        :param weight: Optional. The weight to be set (integer between 0 and
            1). If None, no weight attribute is set. If the value is outside
            the accepted range, it is set to 1.
        :param http_role: Optional. The http_role to be set. This accepts any
            string. Currently, Handle System can process 'conneg'. In future,
            it may be able to process 'no_conneg' and 'browser'.
        :param handle: Optional. Only for the exception message.
        :param all others: Optional. All other key-value pairs will be set to
            the element. Any value is accepted and transformed to string.
        :raise: GenericHandleError: If several 10320/LOC exist (unlikely).

        '''

        # Find existing 10320/LOC entry or create new
        indices = self.__get_python_indices_for_key('10320/LOC', list_of_entries)
        makenew = False
        entry = None
        if len(indices) == 0:
            index = self.__make_another_index(list_of_entries)
            entry = self.__create_entry('10320/LOC', 'add_later', index)
            makenew = True
        else:
            if len(indices) > 1:
                msg = 'There is ' + str(len(indices)) + ' 10320/LOC entries.'
                raise BrokenHandleRecordException(handle=handle, msg=msg)
            ind = indices[0]
            entry = list_of_entries.pop(ind)

        # Get xml data or make new:
        xmlroot = None
        if makenew:
            xmlroot = ET.Element('locations')
            if self.__10320LOC_chooseby is not None:
                xmlroot.set('chooseby', self.__10320LOC_chooseby)
        else:
            try:
                xmlroot = ET.fromstring(entry['data']['value'])
            except TypeError:
                xmlroot = ET.fromstring(entry['data'])
        LOGGER.debug("__add_URL_to_10320LOC: xmlroot is (1) " + ET.tostring(xmlroot, encoding=encoding_value))

        # Check if URL already there...
        location_element = None
        existing_location_ids = []
        if not makenew:
            list_of_locations = xmlroot.findall('location')
            for item in list_of_locations:
                try:
                    existing_location_ids.append(int(item.get('id')))
                except TypeError:
                    pass
                if item.get('href') == url:
                    location_element = item
            existing_location_ids.sort()
        # ... if not, add it!
        if location_element is None:
            location_id = 0
            for existing_id in existing_location_ids:
                if location_id == existing_id:
                    location_id += 1
            location_element = ET.SubElement(xmlroot, 'location')
            LOGGER.debug("__add_URL_to_10320LOC: location_element is (1) " + ET.tostring(location_element, encoding=encoding_value) + ', now add id ' + str(location_id))
            location_element.set('id', str(location_id))
            LOGGER.debug("__add_URL_to_10320LOC: location_element is (2) " + ET.tostring(location_element, encoding=encoding_value) + ', now add url ' + str(url))
            location_element.set('href', url)
            LOGGER.debug("__add_URL_to_10320LOC: location_element is (3) " + ET.tostring(location_element, encoding=encoding_value))
            self.__set_or_adapt_10320LOC_attributes(location_element, weight, http_role, **kvpairs)
        # FIXME: If we start adapting the Handle Record by index (instead of
        # overwriting the entire one), be careful to add and/or overwrite!

        # (Re-)Add entire 10320 to entry, add entry to list of entries:
        LOGGER.debug("__add_URL_to_10320LOC: xmlroot is (2) " + ET.tostring(xmlroot, encoding=encoding_value))
        entry['data'] = ET.tostring(xmlroot, encoding=encoding_value)
        list_of_entries.append(entry)