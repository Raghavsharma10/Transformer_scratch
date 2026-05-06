def folder_create(self, foldername=None, parent_key=None,
                      action_on_duplicate=None, mtime=None):
        """folder/create

        http://www.mediafire.com/developers/core_api/1.3/folder/#create
        """
        return self.request('folder/create', QueryParams({
            'foldername': foldername,
            'parent_key': parent_key,
            'action_on_duplicate': action_on_duplicate,
            'mtime': mtime
        }))