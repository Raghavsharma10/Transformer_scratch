def send_handle_put_request(self, **args):
        '''
        Send a HTTP PUT request to the handle server to write either an entire
            handle or to some specified values to an handle record, using the
            requests module.

        :param handle: The handle.
        :param list_of_entries: A list of handle record entries to be written,
         in the format [{"index":xyz, "type":"xyz", "data":"xyz"}] or similar.
        :param indices: Optional. A list of indices to delete. Defaults
         to None (i.e. the entire handle is deleted.). The list can
         contain integers or strings.
        :param overwrite: Optional. Whether the handle should be overwritten
         if it exists already.
        :return: The server's response.
        '''

        # Check if we have write access at all:
        if not self.__has_write_access:
            raise HandleAuthenticationError(msg=self.__no_auth_message)

        # Check args:
        mandatory_args = ['handle', 'list_of_entries']
        optional_args = ['indices', 'op', 'overwrite']
        b2handle.util.add_missing_optional_args_with_value_none(args, optional_args)
        b2handle.util.check_presence_of_mandatory_args(args, mandatory_args)
        handle = args['handle']
        list_of_entries = args['list_of_entries']
        indices = args['indices']
        op = args['op']
        overwrite = args['overwrite'] or False

        # Make necessary values:
        url = self.make_handle_URL(handle, indices, overwrite=overwrite)
        LOGGER.debug('PUT Request to '+url)
        payload = json.dumps({'values':list_of_entries})
        LOGGER.debug('PUT Request payload: '+payload)
        head = self.__get_headers('PUT')
        LOGGER.debug('PUT Request headers: '+str(head))
        veri = self.__HTTPS_verify

        # Send request to server:
        resp = self.__send_put_request_to_server(url, payload, head, veri, handle)
        if b2handle.hsresponses.is_redirect_from_http_to_https(resp):
            resp = self.__resend_put_request_on_302(payload, head, veri, handle, resp)

        # Check response for authentication issues:
        if b2handle.hsresponses.not_authenticated(resp):
            raise HandleAuthenticationError(
                operation=op,
                handle=handle,
                response=resp,
                username=self.__username
            )
        self.__first_request = False
        return resp, payload