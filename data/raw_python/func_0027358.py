def delete_handle_value(self, handle, key):
        '''
        Delete a key-value pair from a handle record. If the key exists more
        than once, all key-value pairs with this key are deleted.

        :param handle: Handle from whose record the entry should be deleted.
        :param key: Key to be deleted. Also accepts a list of keys.
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        '''
        LOGGER.debug('delete_handle_value...')

        # read handle record:
        handlerecord_json = self.retrieve_handle_record_json(handle)
        if handlerecord_json is None:
            msg = 'Cannot modify unexisting handle'
            raise HandleNotFoundException(handle=handle, msg=msg)
        list_of_entries = handlerecord_json['values']


        # find indices to delete:
        keys = None
        indices = []
        if type(key) != type([]):
            keys = [key]
        else:
            keys = key
        keys_done = []
        for key in keys:

            # filter HS_ADMIN
            if key == 'HS_ADMIN':
                op = 'deleting "HS_ADMIN"'
                raise IllegalOperationException(operation=op, handle=handle)

            if key not in keys_done:
                indices_onekey = self.get_handlerecord_indices_for_key(key, list_of_entries)
                indices = indices + indices_onekey
                keys_done.append(key)

        # Important: If key not found, do not continue, as deleting without indices would delete the entire handle!!
        if not len(indices) > 0:
            LOGGER.debug('delete_handle_value: No values for key(s) ' + str(keys))
            return None
        else:

            # delete and process response:
            op = 'deleting "' + str(keys) + '"'
            resp = self.__send_handle_delete_request(handle, indices=indices, op=op)
            if hsresponses.handle_success(resp):
                LOGGER.debug("delete_handle_value: Deleted handle values " + str(keys) + "of handle " + handle)
            elif hsresponses.values_not_found(resp):
                pass
            else:
                raise GenericHandleError(
                    operation=op,
                    handle=handle,
                    response=resp
                )