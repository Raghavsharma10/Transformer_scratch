def send_handle_delete_request(self, **args):
        '''
        Send a HTTP DELETE request to the handle server to delete either an
            entire handle or to some specified values from a handle record,
            using the requests module.

        :param handle: The handle.
        :param indices: Optional. A list of indices to delete. Defaults to
            None (i.e. the entire handle is deleted.). The list can contain
            integers or strings.
        :return: The server's response.
        '''

        # Check if we have write access at all:
        if not self.__has_write_access:
            raise HandleAuthenticationError(msg=self.__no_auth_message)

        # Check args:
        mandatory_args = ['handle']
        optional_args = ['indices', 'op']
        b2handle.util.add_missing_optional_args_with_value_none(args, optional_args)
        b2handle.util.check_presence_of_mandatory_args(args, mandatory_args)
        handle = args['handle']
        indices = args['indices']
        op = args['op']

        # Make necessary values:
        url = self.make_handle_URL(handle, indices)
        if indices is not None and len(indices) > 0:
            LOGGER.debug('__send_handle_delete_request: Deleting values '+str(indices)+' from handle '+handle+'.')
        else:
            LOGGER.debug('__send_handle_delete_request: Deleting handle '+handle+'.')
        LOGGER.debug('DELETE Request to '+url)
        head = self.__get_headers('DELETE')
        veri = self.__HTTPS_verify

        # Make request:
        resp = None
        if self.__authentication_method == self.__auth_methods['user_pw']:
            resp = self.__session.delete(url, headers=head, verify=veri)
        elif self.__authentication_method == self.__auth_methods['cert']:
            resp = self.__session.delete(url, headers=head, verify=veri, cert=self.__cert_object)
        self.__log_request_response_to_file(
            logger=REQUESTLOGGER,
            op='DELETE',
            handle=handle,
            url=url,
            headers=head,
            verify=veri,
            resp=resp
        )

        # Check response for authentication issues:
        if b2handle.hsresponses.not_authenticated(resp):
            raise HandleAuthenticationError(
                operation=op,
                handle=handle,
                response=resp,
                username=self.__username
            )
        self.__first_request = False
        return resp