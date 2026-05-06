def __send_handle_put_request(self, handle, list_of_entries, indices=None, overwrite=False, op=None):
        '''
        Send a HTTP PUT request to the handle server to write either an entire
            handle or to some specified values to an handle record, using the
            requests module.

        :param handle: The handle.
        :param list_of_entries: A list of handle record entries to be written,
         in the format [{"index":xyz, "type":"xyz", "data":"xyz"}] or similar.
        :param indices: Optional. A list of indices to modify. Defaults
         to None (i.e. the entire handle is updated.). The list can
         contain integers or strings.
        :param overwrite: Optional. Whether the handle should be overwritten
         if it exists already.
        :return: The server's response.
        '''

        resp, payload = self.__handlesystemconnector.send_handle_put_request(
            handle=handle,
            list_of_entries=list_of_entries,
            indices=indices,
            overwrite=overwrite,
            op=op
        )
        return resp, payload