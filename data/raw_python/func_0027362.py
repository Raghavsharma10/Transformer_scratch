def remove_additional_URL(self, handle, *urls):
        '''
        Remove a URL from the handle record's 10320/LOC entry.

        :param handle: The handle to modify.
        :param urls: The URL(s) to be removed. Several URLs may be specified.
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        '''

        LOGGER.debug('remove_additional_URL...')

        handlerecord_json = self.retrieve_handle_record_json(handle)
        if handlerecord_json is None:
            msg = 'Cannot remove URLs from unexisting handle'
            raise HandleNotFoundException(handle=handle, msg=msg)
        list_of_entries = handlerecord_json['values']

        for url in urls:
            self.__remove_URL_from_10320LOC(url, list_of_entries, handle)

        op = 'removing URLs'
        resp, put_payload = self.__send_handle_put_request(
            handle,
            list_of_entries,
            overwrite=True,
            op=op
        )
        # TODO FIXME (one day): Implement overwriting by index (less risky),
        # once HS have fixed the issue with the indices.
        if hsresponses.handle_success(resp):
            pass
        else:
            op = 'removing "' + str(urls) + '"'
            msg = 'Could not remove URLs ' + str(urls)
            raise GenericHandleError(
                operation=op,
                handle=handle,
                reponse=resp,
                msg=msg,
                payload=put_payload
            )