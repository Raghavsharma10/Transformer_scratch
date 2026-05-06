def __create_entry(self, entrytype, data, index, ttl=None):
        '''
        Create an entry of any type except HS_ADMIN.

        :param entrytype: THe type of entry to create, e.g. 'URL' or
            'checksum' or ... Note: For entries of type 'HS_ADMIN', please
            use __create_admin_entry(). For type '10320/LOC', please use
            'add_additional_URL()'
        :param data: The actual value for the entry. Can be a simple string,
            e.g. "example", or a dict {"format":"string", "value":"example"}.
        :param index: The integer to be used as index.
        :param ttl: Optional. If not set, the library's default is set. If
            there is no default, it is not set by this library, so Handle
            System sets it.
        :return: The entry as a dict.
        '''

        if entrytype == 'HS_ADMIN':
            op = 'creating HS_ADMIN entry'
            msg = 'This method can not create HS_ADMIN entries.'
            raise IllegalOperationException(operation=op, msg=msg)

        entry = {'index':index, 'type':entrytype, 'data':data}

        if ttl is not None:
            entry['ttl'] = ttl

        return entry