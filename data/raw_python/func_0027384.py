def send_handle_get_request(self, handle, indices=None):
        '''
        Send a HTTP GET request to the handle server to read either an entire
            handle or to some specified values from a handle record, using the
            requests module.

        :param handle: The handle.
        :param indices: Optional. A list of indices to delete. Defaults to
            None (i.e. the entire handle is deleted.). The list can contain
            integers or strings.
        :return: The server's response.
        '''


        # Assemble required info:
        url = self.make_handle_URL(handle, indices)
        LOGGER.debug('GET Request to '+url)
        head = self.__get_headers('GET')
        veri = self.__HTTPS_verify

        # Send the request
        if self.__cert_needed_for_get_request():
            # If this is the first request and the connector uses client cert authentication, we need to send the cert along
            # in the first request that builds the session.
            resp = self.__session.get(url, headers=head, verify=veri, cert=self.__cert_object)
        else:
            # Normal case:
            resp = self.__session.get(url, headers=head, verify=veri)
    
        # Log and return
        self.__log_request_response_to_file(
            logger=REQUESTLOGGER,
            op='GET',
            handle=handle,
            url=url,
            headers=head,
            verify=veri,
            resp=resp
            )
        self.__first_request = False
        return resp