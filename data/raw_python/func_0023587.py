def file_update_file(self, quick_key, file_extension=None, filename=None,
                         description=None, mtime=None, privacy=None,
                         timezone=None):
        """file/update_file

        http://www.mediafire.com/developers/core_api/1.3/file/#update_file
        """
        return self.request('file/update', QueryParams({
            'quick_key': quick_key,
            'file_extension': file_extension,
            'filename': filename,
            'description': description,
            'mtime': mtime,
            'privacy': privacy,
            'timezone': timezone
        }))