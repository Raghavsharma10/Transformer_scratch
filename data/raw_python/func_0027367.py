def __send_handle_delete_request(self, handle, indices=None, op=None):
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

        resp = self.__handlesystemconnector.send_handle_delete_request(
            handle=handle,
            indices=indices,
            op=op)
        return resp