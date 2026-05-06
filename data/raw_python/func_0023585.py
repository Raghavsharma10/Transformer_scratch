def upload_instant(self, filename, size, hash_, quick_key=None,
                       folder_key=None, filedrop_key=None, path=None,
                       action_on_duplicate=None, mtime=None,
                       version_control=None, previous_hash=None):
        """upload/instant

        http://www.mediafire.com/developers/core_api/1.3/upload/#instant
        """
        return self.request('upload/instant', QueryParams({
            'filename': filename,
            'size': size,
            'hash': hash_,
            'quick_key': quick_key,
            'folder_key': folder_key,
            'filedrop_key': filedrop_key,
            'path': path,
            'action_on_duplicate': action_on_duplicate,
            'mtime': mtime,
            'version_control': version_control,
            'previous_hash': previous_hash
        }))