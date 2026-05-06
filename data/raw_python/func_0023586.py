def file_update(self, quick_key, filename=None, description=None,
                    mtime=None, privacy=None):
        """file/update

        http://www.mediafire.com/developers/core_api/1.3/file/#update
        """
        return self.request('file/update', QueryParams({
            'quick_key': quick_key,
            'filename': filename,
            'description': description,
            'mtime': mtime,
            'privacy': privacy
        }))