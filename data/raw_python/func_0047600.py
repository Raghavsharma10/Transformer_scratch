def get_data(self):
        """Gets the asset content data.

        return: (osid.transport.DataInputStream) - the length of the
                content data
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        # gets you a file-like object...not sure if it will behave
        exactly as expected...
        """
        # read the file from self.get_url()
        # return the file object to be streamed?
        url = self._payload.get_url()
        file_handle = codecs.open(url, 'r', encoding='utf-8')
        try:
            file_handle.read()
        except UnicodeDecodeError:
            file_handle.close()
            # non-Unicode file, like an image
            file_handle = open(url, 'rb')
        file_handle.seek(0)
        return file_handle