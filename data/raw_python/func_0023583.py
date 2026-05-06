def upload_simple(self, fd, filename, folder_key=None, path=None,
                      filedrop_key=None, action_on_duplicate=None,
                      mtime=None, file_size=None, file_hash=None):
        """upload/simple

        http://www.mediafire.com/developers/core_api/1.3/upload/#simple
        """
        action = 'upload/simple'

        params = QueryParams({
            'folder_key': folder_key,
            'path': path,
            'filedrop_key': filedrop_key,
            'action_on_duplicate': action_on_duplicate,
            'mtime': mtime
        })

        headers = QueryParams({
            'X-Filesize': str(file_size),
            'X-Filehash': file_hash,
            'X-Filename': filename.encode('utf-8')
        })

        upload_info = {
            "fd": fd,
        }

        return self.request(action, params, action_token_type="upload",
                            upload_info=upload_info, headers=headers)