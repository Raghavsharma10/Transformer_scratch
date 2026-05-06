def modify_handle_value(self, handle, ttl=None, add_if_not_exist=True, **kvpairs):
        '''
        Modify entries (key-value-pairs) in a handle record. If the key
        does not exist yet, it is created.

        *Note:* We assume that a key exists only once. In case a key exists
        several time, an exception will be raised.

        *Note:* To modify 10320/LOC, please use :meth:`~b2handle.handleclient.EUDATHandleClient.add_additional_URL` or
        :meth:`~b2handle.handleclient.EUDATHandleClient.remove_additional_URL`.

        :param handle: Handle whose record is to be modified
        :param ttl: Optional. Integer value. If ttl should be set to a
            non-default value.
        :param all other args: The user can specify several key-value-pairs.
            These will be the handle value types and values that will be
            modified. The keys are the names or the handle value types (e.g.
            "URL"). The values are the new values to store in "data". If the
            key is 'HS_ADMIN', the new value needs to be of the form
            {'handle':'xyz', 'index':xyz}. The permissions will be set to the
            default permissions.
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        '''
        LOGGER.debug('modify_handle_value...')

        # Read handle record:
        handlerecord_json = self.retrieve_handle_record_json(handle)
        if handlerecord_json is None:
            msg = 'Cannot modify unexisting handle'
            raise HandleNotFoundException(handle=handle, msg=msg)
        list_of_entries = handlerecord_json['values']

        # HS_ADMIN
        if 'HS_ADMIN' in kvpairs.keys() and not self.__modify_HS_ADMIN:
            msg = 'You may not modify HS_ADMIN'
            raise IllegalOperationException(
                msg=msg,
                operation='modifying HS_ADMIN',
                handle=handle
            )

        nothingchanged = True
        new_list_of_entries = []
        list_of_old_and_new_entries = list_of_entries[:]
        keys = kvpairs.keys()
        for key, newval in kvpairs.items():
            # Change existing entry:
            changed = False
            for i in xrange(len(list_of_entries)):
                if list_of_entries[i]['type'] == key:
                    if not changed:
                        list_of_entries[i]['data'] = newval
                        list_of_entries[i].pop('timestamp')  # will be ignored anyway
                        if key == 'HS_ADMIN':
                            newval['permissions'] = self.__HS_ADMIN_permissions
                            list_of_entries[i].pop('timestamp')  # will be ignored anyway
                            list_of_entries[i]['data'] = {
                                'format':'admin',
                                'value':newval
                            }
                            LOGGER.info('Modified' + \
                                ' "HS_ADMIN" of handle ' + handle)
                        changed = True
                        nothingchanged = False
                        new_list_of_entries.append(list_of_entries[i])
                        list_of_old_and_new_entries.append(list_of_entries[i])
                    else:
                        msg = 'There is several entries of type "' + key + '".' + \
                            ' This can lead to unexpected behaviour.' + \
                            ' Please clean up before modifying the record.'
                        raise BrokenHandleRecordException(handle=handle, msg=msg)

            # If the entry doesn't exist yet, add it:
            if not changed:
                if add_if_not_exist:
                    LOGGER.debug('modify_handle_value: Adding entry "' + key + '"' + \
                        ' to handle ' + handle)
                    index = self.__make_another_index(list_of_old_and_new_entries)
                    entry_to_add = self.__create_entry(key, newval, index, ttl)
                    new_list_of_entries.append(entry_to_add)
                    list_of_old_and_new_entries.append(entry_to_add)
                    changed = True
                    nothingchanged = False

        # Add the indices
        indices = []
        for i in xrange(len(new_list_of_entries)):
            indices.append(new_list_of_entries[i]['index'])

        # append to the old record:
        if nothingchanged:
            LOGGER.debug('modify_handle_value: There was no entries ' + \
                str(kvpairs.keys()) + ' to be modified (handle ' + handle + ').' + \
                ' To add them, set add_if_not_exist = True')
        else:
            op = 'modifying handle values'
            resp, put_payload = self.__send_handle_put_request(
                handle,
                new_list_of_entries,
                indices=indices,
                overwrite=True,
                op=op)
            if hsresponses.handle_success(resp):
                LOGGER.info('Handle modified: ' + handle)
            else:
                msg = 'Values: ' + str(kvpairs)
                raise GenericHandleError(
                    operation=op,
                    handle=handle,
                    response=resp,
                    msg=msg,
                    payload=put_payload
                )