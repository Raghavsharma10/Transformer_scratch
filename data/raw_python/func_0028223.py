def attach_files(self, files: Sequence[AttachedFile]):
        '''
        Attach a list of files represented as AttachedFile.
        '''
        assert not self._content, 'content must be empty to attach files.'
        self.content_type = 'multipart/form-data'
        self._attached_files = files