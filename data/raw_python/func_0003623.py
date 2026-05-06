def process_content(self, content, filename=None, content_type=None):
        """Standard implementation of :meth:`.DepotFileInfo.process_content`

        This is the standard depot implementation of files upload, it will
        store the file on the default depot and will provide the standard
        attributes.

        Subclasses will need to call this method to ensure the standard
        set of attributes is provided.
        """

        file_path, file_id = self.store_content(content, filename, content_type)
        self['file_id'] = file_id
        self['path'] = file_path

        saved_file = self.file
        self['filename'] = saved_file.filename
        self['content_type'] = saved_file.content_type
        self['uploaded_at'] = saved_file.last_modified.strftime('%Y-%m-%d %H:%M:%S')
        self['_public_url'] = saved_file.public_url