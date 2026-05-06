def folder_update(self, folder_key, foldername=None, description=None,
                      privacy=None, privacy_recursive=None, mtime=None):
        """folder/update

        http://www.mediafire.com/developers/core_api/1.3/folder/#update
        """
        return self.request('folder/update', QueryParams({
            'folder_key': folder_key,
            'foldername': foldername,
            'description': description,
            'privacy': privacy,
            'privacy_recursive': privacy_recursive,
            'mtime': mtime
        }))