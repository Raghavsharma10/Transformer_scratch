def __get_handle_record_if_necessary(self, handle, handlerecord_json):
        '''
        Returns the handle record if it is None or if its handle is not the
            same as the specified handle.

        '''
        if handlerecord_json is None:
            handlerecord_json = self.retrieve_handle_record_json(handle)
        else:
            if handle != handlerecord_json['handle']:
                handlerecord_json = self.retrieve_handle_record_json(handle)
        return handlerecord_json