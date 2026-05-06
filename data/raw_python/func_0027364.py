def search_handle(self, URL=None, prefix=None, **key_value_pairs):
        '''
        Search for handles containing the specified key with the specified
        value. The search terms are passed on to the reverse lookup servlet
        as-is. The servlet is supposed to be case-insensitive, but if it
        isn't, the wrong case will cause a :exc:`~b2handle.handleexceptions.ReverseLookupException`.

        *Note:* If allowed search keys are configured, only these are used. If
        no allowed search keys are specified, all key-value pairs are
        passed on to the reverse lookup servlet, possibly causing a
        :exc:`~b2handle.handleexceptions.ReverseLookupException`.

        Example calls:

            .. code:: python

                list_of_handles = search_handle('http://www.foo.com')
                list_of_handles = search_handle('http://www.foo.com', CHECKSUM=99999)
                list_of_handles = search_handle(URL='http://www.foo.com', CHECKSUM=99999)


        :param URL: Optional. The URL to search for (reverse lookup). [This is
            NOT the URL of the search servlet!]
        :param prefix: Optional. The Handle prefix to which the search should
            be limited to. If unspecified, the method will search across all
            prefixes present at the server given to the constructor.
        :param key_value_pairs: Optional. Several search fields and values can
            be specified as key-value-pairs,
            e.g. CHECKSUM=123456, URL=www.foo.com
        :raise: :exc:`~b2handle.handleexceptions.ReverseLookupException`: If a search field is specified that
            cannot be used, or if something else goes wrong.
        :return: A list of all Handles (strings) that bear the given key with
            given value of given prefix or server. The list may be empty and
            may also contain more than one element.
        '''
        LOGGER.debug('search_handle...')

        list_of_handles = self.__searcher.search_handle(URL=URL, prefix=prefix, **key_value_pairs)

        return list_of_handles