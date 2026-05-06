def set_content(self, value: RequestContent, *,
                    content_type: str = None):
        '''
        Sets the content of the request.
        '''
        assert self._attached_files is None, \
               'cannot set content because you already attached files.'
        guessed_content_type = 'application/octet-stream'
        if value is None:
            guessed_content_type = 'text/plain'
            self._content = b''
        elif isinstance(value, str):
            guessed_content_type = 'text/plain'
            self._content = value.encode('utf-8')
        else:
            guessed_content_type = 'application/octet-stream'
            self._content = value
        self.content_type = (content_type if content_type is not None
                             else guessed_content_type)