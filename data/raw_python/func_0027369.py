def __send_handle_get_request(self, handle, indices=None):
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

        resp = self.__handlesystemconnector.send_handle_get_request(handle, indices)
        return resp