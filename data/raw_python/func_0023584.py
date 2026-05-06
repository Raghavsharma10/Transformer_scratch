def upload_resumable(self, fd, filesize, filehash, unit_hash, unit_id,
                         unit_size, quick_key=None, action_on_duplicate=None,
                         mtime=None, version_control=None, folder_key=None,
                         filedrop_key=None, path=None, previous_hash=None):
        """upload/resumable

        http://www.mediafire.com/developers/core_api/1.3/upload/#resumable
        """
        action = 'upload/resumable'

        headers = {
            'x-filesize': str(filesize),
            'x-filehash': filehash,
            'x-unit-hash': unit_hash,
            'x-unit-id': str(unit_id),
            'x-unit-size': str(unit_size)
        }

        params = QueryParams({
            'quick_key': quick_key,
            'action_on_duplicate': action_on_duplicate,
            'mtime': mtime,
            'version_control': version_control,
            'folder_key': folder_key,
            'filedrop_key': filedrop_key,
            'path': path,
            'previous_hash': previous_hash
        })

        upload_info = {
            "fd": fd,
            "filename": "chunk"
        }

        return self.request(action, params, action_token_type="upload",
                            upload_info=upload_info, headers=headers)