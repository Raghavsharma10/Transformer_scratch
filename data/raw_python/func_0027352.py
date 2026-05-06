def retrieve_handle_record_json(self, handle):
        '''
        Retrieve a handle record from the Handle server as a complete nested
        dict (including index, ttl, timestamp, ...) for later use.

        Note: For retrieving a simple dict with only the keys and values,
        please use :meth:`~b2handle.handleclient.EUDATHandleClient.retrieve_handle_record`.

        :param handle: The Handle whose record to retrieve.
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :return: The handle record as a nested dict. If the handle does not
            exist, returns None.
        '''
        LOGGER.debug('retrieve_handle_record_json...')

        utilhandle.check_handle_syntax(handle)
        response = self.__send_handle_get_request(handle)
        response_content = decoded_response(response)
                    
        if hsresponses.handle_not_found(response):
            return None
        elif hsresponses.does_handle_exist(response):
         
            handlerecord_json = json.loads(response_content)
            if not handlerecord_json['handle'] == handle:
                raise GenericHandleError(
                    operation='retrieving handle record',
                    handle=handle,
                    response=response,
                    custom_message='The retrieve returned a different handle than was asked for.'
                )
            return handlerecord_json
        elif hsresponses.is_handle_empty(response):
            handlerecord_json = json.loads(response_content)
            return handlerecord_json
        else:
            raise GenericHandleError(
                operation='retrieving',
                handle=handle,
                response=response
            )